//! Track 1 — deterministic Indicator-of-Compromise filter.
//!
//! Cybersecurity IoCs (CVE IDs, IP addresses, file hashes, domains, URLs,
//! registry keys) are short, regular, and have hard equality semantics. We
//! do not need an LLM to evaluate them: regex extraction plus set
//! difference against a caller-owned baseline.
//!
//! This track is deliberately model-free so it can run in CI without
//! credentials, network, or a local daemon.

use std::collections::HashSet;
use std::sync::RwLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::types::Document;

/// Indicator-of-Compromise type. Marked `#[non_exhaustive]` so adding
/// e.g. `Email` or `BitcoinAddress` is non-breaking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum IocKind {
    /// CVE identifier (e.g. `CVE-2024-12345`).
    Cve,
    /// IPv4 address (dotted-decimal).
    Ipv4,
    /// IPv6 address (colon-separated hextets).
    Ipv6,
    /// 32-hex-character MD5 digest.
    Md5,
    /// 40-hex-character SHA-1 digest.
    Sha1,
    /// 64-hex-character SHA-256 digest.
    Sha256,
    /// Fully-qualified domain name.
    Domain,
    /// `http(s)://` URL.
    Url,
    /// Windows registry key (`HKLM\…`, `HKCU\…`).
    RegistryKey,
}

/// A single extracted IoC. Equality is by `(kind, value)` after
/// extractor-side normalisation (lower-casing for domains/URLs/hashes,
/// upper-casing for CVE IDs).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Ioc {
    /// What kind of IoC this is.
    pub kind: IocKind,
    /// Canonical string form. Already normalised by the extractor.
    pub value: String,
}

impl Ioc {
    /// Construct a new IoC with the given kind and (pre-normalised) value.
    pub fn new(kind: IocKind, value: impl Into<String>) -> Self {
        Self {
            kind,
            value: value.into(),
        }
    }
}

/// Extracts IoCs from a [`Document`].
///
/// Implementations must be deterministic: the same input must produce the
/// same set of IoCs across runs. This is what lets Track 1 stand in for an
/// LLM-backed filter in CI gates.
pub trait IocExtractor {
    /// Extract every IoC the implementation recognises. Order is not
    /// significant; the pipeline deduplicates before baseline lookup.
    fn extract(&self, doc: &Document) -> Vec<Ioc>;
}

/// Caller-owned oracle that answers "has this IoC been seen before".
///
/// The trait is async so production implementations can hit a Postgres /
/// Redis / object-store IoC index. The in-memory test impl
/// ([`InMemoryIocBaseline`]) resolves synchronously.
pub trait IocBaseline: Send + Sync {
    /// `true` if `ioc` is already known to the baseline.
    fn contains(&self, ioc: &Ioc) -> impl std::future::Future<Output = Result<bool>> + Send;
}

/// In-memory [`IocBaseline`] backed by a `HashSet`. Suitable for tests,
/// examples, and small embedded deployments.
///
/// Constructed empty; insert known IoCs with [`InMemoryIocBaseline::insert`]
/// or [`InMemoryIocBaseline::extend`]. The set is held behind a `RwLock`
/// so concurrent `contains` calls do not serialise.
#[derive(Debug, Default)]
pub struct InMemoryIocBaseline {
    inner: RwLock<HashSet<Ioc>>,
}

impl InMemoryIocBaseline {
    /// Create an empty baseline.
    pub fn new() -> Self {
        Self::default()
    }

    /// Seed the baseline from an iterator of IoCs. Returns `self` for
    /// builder-style chaining.
    pub fn with_iocs<I>(self, iocs: I) -> Result<Self>
    where
        I: IntoIterator<Item = Ioc>,
    {
        self.extend(iocs)?;
        Ok(self)
    }

    /// Add a single IoC.
    pub fn insert(&self, ioc: Ioc) -> Result<()> {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| Error::Ingestion("ioc baseline lock poisoned".into()))?;
        guard.insert(ioc);
        Ok(())
    }

    /// Bulk-insert IoCs.
    pub fn extend<I>(&self, iocs: I) -> Result<()>
    where
        I: IntoIterator<Item = Ioc>,
    {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| Error::Ingestion("ioc baseline lock poisoned".into()))?;
        guard.extend(iocs);
        Ok(())
    }

    /// Current size. Primarily useful for diagnostics and tests.
    pub fn len(&self) -> Result<usize> {
        let guard = self
            .inner
            .read()
            .map_err(|_| Error::Ingestion("ioc baseline lock poisoned".into()))?;
        Ok(guard.len())
    }

    /// `true` if the baseline holds no IoCs.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
}

impl IocBaseline for InMemoryIocBaseline {
    fn contains(&self, ioc: &Ioc) -> impl std::future::Future<Output = Result<bool>> + Send {
        // Scope the lock guard tightly: no `.await` while it is held,
        // satisfying `clippy::await_holding_lock`.
        let result = (|| -> Result<bool> {
            let guard = self
                .inner
                .read()
                .map_err(|_| Error::Ingestion("ioc baseline lock poisoned".into()))?;
            Ok(guard.contains(ioc))
        })();
        async move { result }
    }
}

/// Default regex-driven [`IocExtractor`]. Recognises CVE IDs, IPv4/IPv6
/// addresses, MD5/SHA-1/SHA-256 hashes, domains, `http(s)` URLs, and
/// Windows registry keys.
///
/// Hosts that need MITRE-aware NER or defanged-IoC handling should
/// implement [`IocExtractor`] themselves; this default exists to keep the
/// happy path zero-config.
#[derive(Debug)]
pub struct RegexIocExtractor {
    cve: Regex,
    ipv4: Regex,
    ipv6: Regex,
    md5: Regex,
    sha1: Regex,
    sha256: Regex,
    domain: Regex,
    url: Regex,
    reg_key: Regex,
}

impl RegexIocExtractor {
    /// Build the default extractor. Returns an error only if one of the
    /// baked-in patterns fails to compile; `tests/ingestion_ioc.rs`
    /// exercises this path.
    pub fn new() -> Result<Self> {
        Ok(Self {
            cve: compile(r"\bCVE-\d{4}-\d{4,7}\b")?,
            ipv4: compile(
                r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b",
            )?,
            ipv6: compile(r"\b(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}\b")?,
            md5: compile(r"\b[A-Fa-f0-9]{32}\b")?,
            sha1: compile(r"\b[A-Fa-f0-9]{40}\b")?,
            sha256: compile(r"\b[A-Fa-f0-9]{64}\b")?,
            domain: compile(
                r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b",
            )?,
            url: compile(r#"https?://[^\s<>\[\]\(\)\{\}\\"',]+"#)?,
            reg_key: compile(r"\bHK(?:LM|CU|CR|U|CC)\\[^\s,;]+")?,
        })
    }

    fn collect(&self, text: &str, out: &mut Vec<Ioc>) {
        // URLs first so we can filter out domains nested inside.
        let mut url_spans = Vec::new();
        for m in self.url.find_iter(text) {
            url_spans.push((m.start(), m.end()));
            out.push(Ioc::new(IocKind::Url, m.as_str().to_ascii_lowercase()));
        }

        for m in self.cve.find_iter(text) {
            out.push(Ioc::new(IocKind::Cve, m.as_str().to_ascii_uppercase()));
        }
        for m in self.ipv4.find_iter(text) {
            out.push(Ioc::new(IocKind::Ipv4, m.as_str().to_string()));
        }
        for m in self.ipv6.find_iter(text) {
            out.push(Ioc::new(IocKind::Ipv6, m.as_str().to_ascii_lowercase()));
        }
        // Longest-first; word boundaries already prevent a 32-char prefix
        // of a SHA-256 from being captured as an MD5, but be explicit.
        for m in self.sha256.find_iter(text) {
            out.push(Ioc::new(IocKind::Sha256, m.as_str().to_ascii_lowercase()));
        }
        for m in self.sha1.find_iter(text) {
            out.push(Ioc::new(IocKind::Sha1, m.as_str().to_ascii_lowercase()));
        }
        for m in self.md5.find_iter(text) {
            out.push(Ioc::new(IocKind::Md5, m.as_str().to_ascii_lowercase()));
        }
        for m in self.domain.find_iter(text) {
            let (start, end) = (m.start(), m.end());
            // Drop domain matches that fall inside an already-captured URL.
            let inside_url = url_spans.iter().any(|&(s, e)| start >= s && end <= e);
            if inside_url {
                continue;
            }
            out.push(Ioc::new(IocKind::Domain, m.as_str().to_ascii_lowercase()));
        }
        for m in self.reg_key.find_iter(text) {
            out.push(Ioc::new(IocKind::RegistryKey, m.as_str().to_string()));
        }
    }
}

impl IocExtractor for RegexIocExtractor {
    fn extract(&self, doc: &Document) -> Vec<Ioc> {
        let mut out = Vec::new();
        self.collect(&doc.text, &mut out);
        for section in &doc.sections {
            self.collect(&section.text, &mut out);
        }
        // Deduplicate within the document — set difference against the
        // baseline happens in the pipeline.
        let mut seen: HashSet<Ioc> = HashSet::with_capacity(out.len());
        out.retain(|i| seen.insert(i.clone()));
        out
    }
}

fn compile(pattern: &str) -> Result<Regex> {
    Regex::new(pattern).map_err(|err| {
        Error::Ingestion(format!(
            "RegexIocExtractor: pattern {pattern:?} failed to compile: {err}"
        ))
    })
}
