//! Estimated highlight spans for a search result.
//!
//! Vector and literal searches return raw OCR text without match positions. This module estimates
//! display-only exact, stem, prefix, and fuzzy spans. [`MatchKind`] distinguishes estimates from
//! index-reported matches.

/// Minimum query length for prefix matching.
const MIN_PREFIX_LEN: usize = 3;
/// Minimum token and query length for one-edit fuzzy matching.
const MIN_FUZZY_LEN: usize = 5;

/// Query words excluded from display highlighting.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "how", "in", "into",
    "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "what",
    "when", "which", "with",
];

/// Match confidence, ordered weakest to strongest so merged spans retain the strongest evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MatchKind {
    /// Tokens differ by one edit.
    Fuzzy,
    /// A query term prefixes the token.
    Prefix,
    /// Tokens share a stem.
    Stem,
    /// The tokens are equal once case and punctuation are gone.
    Exact,
    /// Reported by the full-text index and recovered by [`from_marked`].
    /// Never produced by [`highlights`].
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
/// Bounds are Unicode scalar-value offsets, not UTF-8 bytes or JavaScript UTF-16 code units.
/// JavaScript clients can index the same units with `Array.from(text)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Highlight {
    /// Inclusive start, in Unicode scalar values from the beginning of the text.
    pub start: usize,
    /// Exclusive end, in Unicode scalar values.
    pub end: usize,
    /// The strongest evidence found inside the span.
    pub kind: MatchKind,
}

/// Guesses which parts of `text` the words of `query` were about.
///
/// Adjacent matches merge; returned spans are ordered and non-overlapping. Empty and stopword-only
/// queries return no spans.
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
/// The delimiters are ordinary characters that OCR text can contain on its own, so the parse is
/// conservative: unmatched or nested delimiters remain literal text.
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

/// Split on non-alphanumeric Unicode scalar values.
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

/// Crude suffix stripping for display hints; the result need not be a word.
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

/// Whether one insertion, deletion, or substitution turns `left` into `right`.
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
