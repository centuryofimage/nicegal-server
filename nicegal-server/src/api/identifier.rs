use super::error::ApiError;

/// Client-chosen identifiers are echoed into logs and used as map keys, so they stay short and
/// limited to ASCII letters, digits, and `-_:`.
pub(super) fn validate_identifier(
    value: &str,
    max_len: usize,
    message: &'static str,
) -> Result<(), ApiError> {
    if value.is_empty()
        || value.len() > max_len
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_:".contains(&b))
    {
        return Err(ApiError::bad_request(message));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enforces_charset_and_the_callers_length_limit() {
        assert!(validate_identifier("window-1:a_b", 16, "bad").is_ok());
        assert!(validate_identifier("", 16, "bad").is_err());
        assert!(validate_identifier("a b", 16, "bad").is_err());
        assert!(validate_identifier("ü", 16, "bad").is_err());
        assert!(validate_identifier(&"a".repeat(16), 16, "bad").is_ok());
        assert!(validate_identifier(&"a".repeat(17), 16, "bad").is_err());
    }
}
