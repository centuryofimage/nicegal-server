//! File-name and path predicates shared by every search lane.
//!
//! Matching is case-insensitive and treats `\` and `/` as the same separator. A pattern without
//! `*` or `?` is a substring; with either it must match the whole target. A wildcard pattern that
//! contains a separator targets the full path, otherwise the file name, so `IMG_*.jpg` and
//! `*/2024/*.png` both mean what they look like. `*` crosses separators.

/// Which part of a path a pattern is compared against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternTarget {
    Name,
    Path,
}

/// A normalized name or path pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathPattern {
    /// Lowercased, with `/` as the only separator.
    text: String,
    wildcard: bool,
    target: PatternTarget,
}

impl PathPattern {
    /// A `path:` pattern: substrings search the full path, wildcards follow the separator rule.
    pub fn path(pattern: &str) -> Self {
        let text = normalize(pattern);
        let wildcard = has_wildcard(&text);
        let target = if wildcard && !text.contains('/') {
            PatternTarget::Name
        } else {
            PatternTarget::Path
        };
        Self {
            text,
            wildcard,
            target,
        }
    }

    /// A `name:` pattern, always compared against the file name.
    pub fn name(pattern: &str) -> Self {
        let text = normalize(pattern);
        Self {
            wildcard: has_wildcard(&text),
            text,
            target: PatternTarget::Name,
        }
    }

    pub fn target(&self) -> PatternTarget {
        self.target
    }

    /// The normalized pattern text. [`PathPattern::path`] of this is the same pattern.
    pub fn as_str(&self) -> &str {
        &self.text
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// The longest run of literal characters, which every match must contain. Wildcard searches use
    /// it to narrow candidates through the trigram index before the exact comparison.
    pub fn required_literal(&self) -> &str {
        if !self.wildcard {
            return &self.text;
        }
        self.text
            .split(['*', '?'])
            .max_by_key(|run| run.len())
            .unwrap_or_default()
    }

    /// Match an already normalized path.
    pub fn matches_normalized(&self, path: &str) -> bool {
        let subject = match self.target {
            PatternTarget::Name => file_name(path),
            PatternTarget::Path => path,
        };
        if self.wildcard {
            glob_match(&self.text, subject)
        } else {
            subject.contains(&self.text)
        }
    }

    pub fn matches(&self, path: &str) -> bool {
        self.matches_normalized(&normalize(path))
    }
}

/// One filter term. Every term in a search must hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileFilter {
    Path {
        pattern: PathPattern,
        exclude: bool,
    },
    /// Lowercased extensions without the leading dot. Matches when the file has any of them.
    Extension {
        extensions: Vec<String>,
        exclude: bool,
    },
}

impl FileFilter {
    pub fn path(pattern: &str, exclude: bool) -> Self {
        Self::Path {
            pattern: PathPattern::path(pattern),
            exclude,
        }
    }

    pub fn extensions<'a>(extensions: impl IntoIterator<Item = &'a str>, exclude: bool) -> Self {
        Self::Extension {
            extensions: extensions
                .into_iter()
                .map(|extension| extension.trim_start_matches('.').to_lowercase())
                .filter(|extension| !extension.is_empty())
                .collect(),
            exclude,
        }
    }
}

pub(crate) fn normalize(path: &str) -> String {
    path.replace('\\', "/").to_lowercase()
}

fn has_wildcard(text: &str) -> bool {
    text.contains(['*', '?'])
}

fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// The lowercased extension of a path's file name, or an empty string.
pub(crate) fn extension(path: &str) -> String {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    match name.rfind('.') {
        Some(dot) if dot > 0 => name[dot + 1..].to_lowercase(),
        _ => String::new(),
    }
}

/// Whole-string match where `*` is any run of characters and `?` is exactly one.
fn glob_match(pattern: &str, text: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();
    let (mut p, mut t) = (0, 0);
    // The most recent `*` and the text position it currently absorbs up to.
    let mut star: Option<(usize, usize)> = None;
    while t < text.len() {
        match pattern.get(p) {
            Some('*') => {
                star = Some((p, t));
                p += 1;
            }
            Some(&c) if c == '?' || c == text[t] => {
                p += 1;
                t += 1;
            }
            _ => match star {
                Some((star_p, star_t)) => {
                    p = star_p + 1;
                    t = star_t + 1;
                    star = Some((star_p, star_t + 1));
                }
                None => return false,
            },
        }
    }
    pattern[p..].iter().all(|c| *c == '*')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substrings_search_the_full_path() {
        let pattern = PathPattern::path("Trips\\CAFÉ");
        assert!(pattern.matches("D:\\Photos\\trips\\café.jpg"));
        assert!(PathPattern::path("mp4").matches("D:\\clips\\a.MP4"));
        assert!(!PathPattern::path("mp4").matches("D:\\clips\\a.jpg"));
    }

    #[test]
    fn wildcards_without_a_separator_match_the_whole_name() {
        let pattern = PathPattern::path("IMG_*.jpg");
        assert_eq!(pattern.target(), PatternTarget::Name);
        assert!(pattern.matches("D:\\img_dir\\IMG_0001.JPG"));
        assert!(!pattern.matches("D:\\IMG_dir\\photo.jpg"));
        assert!(!pattern.matches("D:\\x\\IMG_0001.jpg.bak"));
        assert!(PathPattern::path("a?c.png").matches("/x/abc.png"));
        assert!(!PathPattern::path("a?c.png").matches("/x/ac.png"));
    }

    #[test]
    fn wildcards_with_a_separator_match_the_whole_path() {
        let pattern = PathPattern::path("*\\2024\\*.png");
        assert_eq!(pattern.target(), PatternTarget::Path);
        assert!(pattern.matches("D:\\Photos\\2024\\trip\\a.png"));
        assert!(!pattern.matches("D:\\Photos\\2025\\a.png"));
        assert_eq!(pattern.required_literal(), "/2024/");
    }

    #[test]
    fn name_patterns_ignore_folders() {
        assert!(!PathPattern::name("photo").matches("/photo folder/other.jpg"));
        assert!(PathPattern::name("*.jpg").matches("/photo folder/other.jpg"));
    }

    #[test]
    fn extensions_ignore_case_and_dots() {
        assert_eq!(extension("D:\\a.b\\Clip.MP4"), "mp4");
        assert_eq!(extension("/a/.hidden"), "");
        assert_eq!(extension("/a/noext"), "");
        assert_eq!(
            FileFilter::extensions([".JPG", "png", ""], false),
            FileFilter::Extension {
                extensions: vec!["jpg".into(), "png".into()],
                exclude: false
            }
        );
    }
}
