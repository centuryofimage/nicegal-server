//! Estimated highlight spans for a search result.
//!
//! Vector search answers *which* chunk is relevant, never *why*: a nearest neighbour has a cosine
//! distance and nothing else to point at. The FTS5 modes get their bracketed excerpt from SQLite,
//! but the vector, glob, and regex modes hand back raw OCR text, so a UI that wants to show a
//! reader what it thinks matched has nothing to underline.
//!
//! This module is that explanation pass, and it is deliberately a cheap lexical one rather than a
//! second index: retrieval has already chosen the result, so all that is left is guessing which
//! words in it the query was about. The guess is graded, strongest first:
//!
//! ```text
//! exact token match   receipts    <-> receipts
//! stem match          inspecting  <-> inspected
//! prefix match        regul       <-> regulations
//! fuzzy match         regulations <-> regulatlons   (one OCR error)
//! ```
//!
//! Exact matching alone is too brittle for OCR output and prefix matching alone is too permissive,
//! which is why the ladder exists — and why a caller gets the [`MatchKind`] back rather than a
//! bare span, so a UI can style a guess differently from a certainty. A wrong highlight costs a
//! reader a glance, so the tuning here leans towards showing something.

/// A query term shorter than this never prefix-matches: two letters would light up half the page.
const MIN_PREFIX_LEN: usize = 3;
/// A token shorter than this never fuzzy-matches, because one edit in a four-letter word is a
/// different word far more often than it is an OCR error.
const MIN_FUZZY_LEN: usize = 5;

/// Words that carry no information about what a picture says, dropped from the query so a search
/// for "the receipt" does not underline every "the" on the page.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "how", "in", "into",
    "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "what",
    "when", "which", "with",
];

/// How a span was matched, in ascending confidence: [`MatchKind::Indexed`] is the strongest.
///
/// The ordering is the point of the type — [`Ord`] is what lets a merged span report the best
/// evidence it contains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MatchKind {
    /// The tokens differ by a single character edit: the OCR-error case.
    Fuzzy,
    /// A query term is a prefix of the token: the partially-typed-word case.
    Prefix,
    /// The tokens share a stem, as "searching" and "searches" do.
    Stem,
    /// The tokens are equal once case and punctuation are gone.
    Exact,
    /// Not a guess at all: the full-text index reported this token as a match, and
    /// [`from_marked`] recovered where it said so. Never produced by [`highlights`].
    ///
    /// It outranks [`MatchKind::Exact`] because it is evidence rather than inference, and it is
    /// kept distinct from it because the two are not the same claim: a prefix query marks
    /// `dresses` for `dre*`, which is a real match and not an equal token.
    Indexed,
}

impl MatchKind {
    /// The wire name, which is also what the HTTP API serialises.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fuzzy => "fuzzy",
            Self::Prefix => "prefix",
            Self::Stem => "stem",
            Self::Exact => "exact",
            Self::Indexed => "indexed",
        }
    }
}

/// A span of the searched text worth highlighting.
///
/// The bounds are **character** offsets, not byte offsets: the text they index into crosses the
/// wire as JSON and gets sliced by a JavaScript client, which cannot act on a UTF-8 byte offset
/// once OCR output contains an accent or a curly quote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Highlight {
    /// Inclusive start, in characters from the beginning of the text.
    pub start: usize,
    /// Exclusive end, in characters.
    pub end: usize,
    /// The strongest evidence found inside the span.
    pub kind: MatchKind,
}

/// Guesses which parts of `text` the words of `query` were about.
///
/// Runs of neighbouring matched words collapse into one span, so a query of "construction
/// regulations" over "... new construction regulations ..." underlines the phrase rather than two
/// words with a gap. Spans come back in reading order and never overlap. A query made entirely of
/// stopwords, or of nothing, highlights nothing.
pub fn highlights(query: &str, text: &str) -> Vec<Highlight> {
    let mut terms: Vec<String> = tokens(query)
        .into_iter()
        .map(|token| token.word)
        .filter(|word| !STOPWORDS.contains(&word.as_str()))
        .collect();
    terms.sort_unstable();
    terms.dedup();
    if terms.is_empty() {
        return Vec::new();
    }
    let stems: Vec<String> = terms.iter().map(|term| stem(term)).collect();

    let mut spans: Vec<Highlight> = Vec::new();
    // The index of the token that produced the last span, so only *adjacent* matches merge: two
    // matches with an unmatched word between them are two separate pieces of evidence.
    let mut previous: Option<usize> = None;
    for (index, token) in tokens(text).into_iter().enumerate() {
        let Some(kind) = classify(&token.word, &terms, &stems) else {
            continue;
        };
        match spans.last_mut() {
            Some(last) if previous.is_some_and(|last_index| last_index + 1 == index) => {
                last.end = token.end;
                last.kind = last.kind.max(kind);
            }
            _ => spans.push(Highlight {
                start: token.start,
                end: token.end,
                kind,
            }),
        }
        previous = Some(index);
    }
    spans
}

/// Turns a snippet that already carries match markers into the plain text and the spans those
/// markers covered, so an FTS5 result and an estimated one reach a client in the same shape.
///
/// FTS5's `snippet()` wraps the terms it matched in a pair of delimiters of the caller's choosing
/// ([`crate::db::SNIPPET_OPEN`] and [`crate::db::SNIPPET_CLOSE`]); that is authoritative — the
/// index is saying which tokens it matched — but it is also a string the client would have to
/// parse. This does that parse once, here.
///
/// The delimiters are ordinary characters that OCR text can contain on its own, so the parse is
/// deliberately conservative: an `open` with no `close` after it, a second `open` inside a marked
/// run, and a `close` with nothing open are all left in the text as the literal characters they
/// probably are, rather than being allowed to swallow the rest of the snippet.
pub fn from_marked(snippet: &str, open: char, close: char) -> (String, Vec<Highlight>) {
    let mut text = String::with_capacity(snippet.len());
    let mut spans = Vec::new();
    // Counted in characters, and only over what survives into `text`, so a span indexes the
    // snippet the client is handed rather than the marked-up one it never sees.
    let mut written = 0;
    let mut opened: Option<usize> = None;
    for (index, character) in snippet.char_indices() {
        if character == open
            && opened.is_none()
            && snippet[index + character.len_utf8()..].contains(close)
        {
            opened = Some(written);
        } else if character == close
            && let Some(start) = opened.take()
        {
            if written > start {
                spans.push(Highlight {
                    start,
                    end: written,
                    kind: MatchKind::Indexed,
                });
            }
        } else {
            text.push(character);
            written += 1;
        }
    }
    (text, spans)
}

/// The best evidence any query term offers for `word`, or `None` if none of them match it.
///
/// The ladder is walked whole rather than short-circuited on the first hit: one term matching by
/// prefix must not hide another matching exactly.
fn classify(word: &str, terms: &[String], stems: &[String]) -> Option<MatchKind> {
    let word_stem = stem(word);
    terms
        .iter()
        .zip(stems)
        .filter_map(|(term, term_stem)| {
            if word == term {
                Some(MatchKind::Exact)
            } else if word_stem == *term_stem {
                Some(MatchKind::Stem)
            } else if term.chars().count() >= MIN_PREFIX_LEN && word.starts_with(term.as_str()) {
                Some(MatchKind::Prefix)
            } else if word.chars().count() >= MIN_FUZZY_LEN
                && term.chars().count() >= MIN_FUZZY_LEN
                && within_one_edit(word, term)
            {
                Some(MatchKind::Fuzzy)
            } else {
                None
            }
        })
        .max()
}

/// One word of the input, lowercased, with the character bounds it occupied in the original.
struct Token {
    word: String,
    start: usize,
    end: usize,
}

/// Splits on everything that is not a letter or a digit, which is the right split for OCR text:
/// punctuation is where the recogniser invents characters most often, and a hyphen or a stray
/// comma inside a phrase should not stop a word from matching.
fn tokens(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut word = String::new();
    let mut start = 0;
    let mut length = 0;
    for (index, character) in text.chars().enumerate() {
        length = index + 1;
        if character.is_alphanumeric() {
            if word.is_empty() {
                start = index;
            }
            word.extend(character.to_lowercase());
        } else if !word.is_empty() {
            tokens.push(Token {
                word: std::mem::take(&mut word),
                start,
                end: index,
            });
        }
    }
    if !word.is_empty() {
        tokens.push(Token {
            word,
            start,
            end: length,
        });
    }
    tokens
}

/// A deliberately crude suffix stripper — enough to join "inspecting" to "inspected" and
/// "regulation" to "regulations" without carrying a full Porter stemmer for a highlight hint.
/// It only ever has to agree with itself, so it may return something that is not a word.
fn stem(word: &str) -> String {
    let length = word.chars().count();
    let mut stemmed = word.to_owned();
    for (suffix, minimum, replacement) in [
        ("ies", 5, "y"),
        ("ches", 6, "ch"),
        ("shes", 6, "sh"),
        ("ing", 6, ""),
        ("edly", 7, ""),
        ("ed", 5, ""),
        ("ly", 5, ""),
        ("es", 5, ""),
        ("s", 4, ""),
    ] {
        // "address" and "glass" keep their tail: the trailing "s" is part of the word.
        if length >= minimum && stemmed.ends_with(suffix) && !stemmed.ends_with("ss") {
            stemmed.truncate(stemmed.len() - suffix.len());
            stemmed.push_str(replacement);
            break;
        }
    }
    // "running" strips to "runn"; dropping the doubled consonant lands it on "run".
    let mut characters = stemmed.chars().rev();
    if let (Some(last), Some(before)) = (characters.next(), characters.next())
        && last == before
        && last.is_alphabetic()
        && !matches!(last, 'a' | 'e' | 'i' | 'o' | 'u' | 's')
    {
        stemmed.pop();
    }
    stemmed
}

/// Whether one insertion, deletion, or substitution turns `left` into `right`. Cheaper than a full
/// edit-distance matrix and all the OCR-error tolerance a highlight needs.
fn within_one_edit(left: &str, right: &str) -> bool {
    let left: Vec<char> = left.chars().collect();
    let right: Vec<char> = right.chars().collect();
    let (long, short) = if left.len() >= right.len() {
        (&left, &right)
    } else {
        (&right, &left)
    };
    if long.len() - short.len() > 1 {
        return false;
    }
    let (mut long_index, mut short_index, mut edits) = (0, 0, 0);
    while long_index < long.len() && short_index < short.len() {
        if long[long_index] == short[short_index] {
            long_index += 1;
            short_index += 1;
            continue;
        }
        edits += 1;
        if edits > 1 {
            return false;
        }
        // Equal lengths means the edit was a substitution, so both sides advance; otherwise it was
        // an insertion in the longer side, and only that side does.
        long_index += 1;
        if long.len() == short.len() {
            short_index += 1;
        }
    }
    edits + (long.len() - long_index) + (short.len() - short_index) <= 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimated_and_indexed_spans_land_on_the_right_words() {
        // The fixtures are ASCII, so slicing them by the character offsets a `Highlight` carries
        // is the same as slicing them by bytes.
        let text = "Notice: regulatlons covering new construction projects, inspected weekly.";
        let found = highlights("construction regulations inspecting", text);
        let seen: Vec<(&str, MatchKind)> = found
            .iter()
            .map(|highlight| (&text[highlight.start..highlight.end], highlight.kind))
            .collect();
        assert_eq!(
            seen,
            vec![
                // One OCR slip (l for i) still explains the hit.
                ("regulatlons", MatchKind::Fuzzy),
                ("construction", MatchKind::Exact),
                // "inspecting" and "inspected" share a stem, while "projects" and "weekly" are
                // words the query never asked about.
                ("inspected", MatchKind::Stem),
            ]
        );

        // Neighbouring matches become one phrase, reported at its best evidence, and a partly
        // typed word still finds the whole one.
        let phrase = "new construction regulations here";
        let merged = highlights("construction regul", phrase);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].kind, MatchKind::Exact);
        assert_eq!(
            &phrase[merged[0].start..merged[0].end],
            "construction regulations"
        );

        // A query with nothing to say about the page underlines nothing.
        assert!(highlights("the and of", text).is_empty());
        assert!(highlights("", text).is_empty());

        // The FTS5 path reaches the same shape from the other direction: markers in, plain text
        // and spans out, offsets counted over the text the client is handed rather than the
        // marked-up one. An unpaired delimiter is OCR text, not a marker, and stays put.
        let (plain, marked) = from_marked("a [coffee] shop [receipt] for [50", '[', ']');
        assert_eq!(plain, "a coffee shop receipt for [50");
        assert_eq!(
            marked
                .iter()
                .map(|highlight| (&plain[highlight.start..highlight.end], highlight.kind))
                .collect::<Vec<_>>(),
            vec![
                ("coffee", MatchKind::Indexed),
                ("receipt", MatchKind::Indexed)
            ]
        );
    }
}
