#!/usr/bin/env node
/**
 * check-audit-allowlist.mjs — reviewed-exception gate for `npm audit` on site/.
 *
 * Ported from mcp-tool-shop-org/brand scripts/check-audit-allowlist.mjs (brand
 * 1f72fa0). The behaviour is brand's: any high/critical advisory not listed in
 * .github/audit-allowlist.json, or listed but past its expiry, fails the build
 * exactly as bare `npm audit --audit-level=high` did. Only a reviewed,
 * time-boxed, git-tracked exception lets one advisory pass while a fix is
 * prepared.
 *
 * Additions in this port, all of them fail-closed:
 *   - The report is always parsed and its shape checked, even when npm exited 0
 *     (an `error` object, a missing `vulnerabilities` object, an unknown
 *     severity or a non-array `via` fails).
 *   - A transitive entry (a `via` that names another package) is judged by the
 *     advisories it resolves to through the report; a name the report does not
 *     list fails. It is never allowed by default.
 *   - An exception applies to an advisory only if its `package` is the package
 *     the advisory is filed under, its `range` equals the advisory's reported
 *     range once all whitespace is removed from both ("<= 4.0.7" == "<=4.0.7"),
 *     and npm reports no fix for that package. So the exception closes on
 *     whichever comes first: its expiry, a change in the advisory's range, or a
 *     fix shipping. Each is the cue to bump the package and remove the entry.
 *   - "No fix" means `fixAvailable` is false or absent, or is npm's semver-major
 *     suggestion to move ANOTHER package to a version LOWER than the one locked
 *     in the package-lock.json given as the fourth argument: npm offering to
 *     downgrade a parent until the vulnerable package drops out, not a fix for
 *     it. `true`, a non-major suggestion, one naming the package itself, an
 *     upgrade, or anything the lockfile cannot settle counts as a fix.
 *   - Dates must be YYYY-MM-DD; an entry missing a field does not apply.
 *
 * Usage:
 *   node scripts/check-audit-allowlist.mjs <audit-report.json> <allowlist.json> <npmAuditExitCode> [package-lock.json]
 *
 * Exit code: 0 if every flagged high/critical advisory is covered by a
 * non-expired, still-matching allowlist entry (or there is none); non-zero
 * otherwise, including when the report or the allowlist cannot be read or
 * verified — ambiguous states are never treated as a pass.
 */
import { readFileSync, existsSync } from 'node:fs';

const [, , auditReportPath, allowlistPath, npmAuditExitCodeArg, lockfilePath] = process.argv;

if (!auditReportPath || !allowlistPath || npmAuditExitCodeArg === undefined) {
  console.error(
    'Usage: node scripts/check-audit-allowlist.mjs <audit-report.json> <allowlist.json> <npmAuditExitCode> [package-lock.json]'
  );
  process.exit(1);
}

if (!/^\d+$/.test(npmAuditExitCodeArg)) {
  console.error(`npmAuditExitCode must be a non-negative integer, got ${JSON.stringify(npmAuditExitCodeArg)}. Failing closed.`);
  process.exit(1);
}
const npmAuditExitCode = Number(npmAuditExitCodeArg);
const failClosed = () => process.exit(npmAuditExitCode || 1);

const SEVERITIES = new Set(['info', 'low', 'moderate', 'high', 'critical']);
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const GHSA_URL_RE = /\/advisories\/(GHSA(?:-[23456789cfghjmpqrvwx]{4}){3})$/;
const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const squash = (s) => s.replace(/\s+/g, '');
const ghsaOf = (advisory) => (typeof advisory.url === 'string' && GHSA_URL_RE.exec(advisory.url)?.[1]) || 'no GHSA id';

let auditReportRaw;
try {
  auditReportRaw = readFileSync(auditReportPath, 'utf-8');
} catch (err) {
  console.error(`Could not read audit report at ${auditReportPath}: ${err.message}`);
  console.error('Failing closed (treating as a real audit failure).');
  failClosed();
}

let auditReport;
try {
  auditReport = JSON.parse(auditReportRaw);
} catch (err) {
  console.error('npm audit did not return parseable JSON — likely a network error or npm-internal');
  console.error('failure rather than a real advisory list. Failing closed.');
  console.error(err.message);
  console.error(auditReportRaw.slice(0, 2000));
  failClosed();
}

function malformed(why) {
  console.error(`npm audit report is not a usable v2 report: ${why}. Failing closed.`);
  console.error(auditReportRaw.slice(0, 2000));
  failClosed();
}

if (!isObject(auditReport)) malformed('it is not a JSON object');
if (auditReport.error !== undefined) malformed(`npm reported an error: ${JSON.stringify(auditReport.error)}`);
if (!isObject(auditReport.vulnerabilities)) malformed('it has no `vulnerabilities` object');
const vulns = auditReport.vulnerabilities;
for (const [name, vuln] of Object.entries(vulns)) {
  if (!isObject(vuln)) malformed(`vulnerabilities.${name} is not an object`);
  if (!SEVERITIES.has(vuln.severity)) malformed(`vulnerabilities.${name}.severity is ${JSON.stringify(vuln.severity)}`);
  if (!Array.isArray(vuln.via) || vuln.via.length === 0) malformed(`vulnerabilities.${name}.via is not a non-empty array`);
  for (const via of vuln.via) {
    if (typeof via !== 'string' && !(isObject(via) && via.source !== undefined)) {
      malformed(`vulnerabilities.${name}.via holds an entry that is neither a package name nor an advisory with a source id`);
    }
  }
}

// The lockfile, when given, settles whether a semver-major fixAvailable is a downgrade.
let lockPackages = null;
if (lockfilePath !== undefined) {
  try {
    lockPackages = JSON.parse(readFileSync(lockfilePath, 'utf-8')).packages;
  } catch (err) {
    console.error(`Could not read the lockfile ${lockfilePath}: ${err.message}. Failing closed.`);
    process.exit(1);
  }
  if (!isObject(lockPackages)) {
    console.error(`${lockfilePath} has no \`packages\` object. Failing closed.`);
    process.exit(1);
  }
}

const VERSION_RE = /^(\d+)\.(\d+)\.(\d+)$/;
const parseVersion = (v) => (typeof v === 'string' ? VERSION_RE.exec(v)?.slice(1).map(Number) : undefined);
function isLower(a, b) {
  for (let i = 0; i < 3; i += 1) if (a[i] !== b[i]) return a[i] < b[i];
  return false;
}

// null when npm reports no fix for `owner`; otherwise a description of the fix.
// `notes` collects the suggestions that were set aside as downgrades.
function fixFor(owner, fixAvailable, notes) {
  if (fixAvailable === undefined || fixAvailable === false) return null;
  if (
    isObject(fixAvailable) &&
    fixAvailable.isSemVerMajor === true &&
    typeof fixAvailable.name === 'string' &&
    fixAvailable.name !== owner
  ) {
    const locked = lockPackages?.[`node_modules/${fixAvailable.name}`]?.version;
    const to = parseVersion(fixAvailable.version);
    const from = parseVersion(locked);
    if (to && from && isLower(to, from)) {
      notes.add(
        `npm's only suggestion is a semver-major downgrade of ${fixAvailable.name} ${locked} -> ${fixAvailable.version}, not a fix for ${owner}`
      );
      return null;
    }
  }
  return JSON.stringify(fixAvailable);
}

// Every advisory a vulnerability resolves to. `via[]` entries are either a
// plain string (naming another vulnerable package this one flows through) or
// an object carrying the actual advisory (source/title/url/range/...). Strings
// are followed through the report; a name it does not list is unresolved.
function resolve(name, seen = new Set()) {
  if (seen.has(name)) return { advisories: [], unresolved: [] };
  seen.add(name);
  const vuln = vulns[name];
  if (!vuln) return { advisories: [], unresolved: [name] };
  const out = { advisories: [], unresolved: [] };
  for (const via of vuln.via) {
    if (typeof via === 'string') {
      const sub = resolve(via, seen);
      out.advisories.push(...sub.advisories);
      out.unresolved.push(...sub.unresolved);
    } else {
      out.advisories.push({ id: String(via.source), owner: name, ownerVuln: vuln, advisory: via });
    }
  }
  return out;
}

// Every advisory (npm's numeric `source` id) reached from a high/critical package.
const flagged = new Map(); // id -> [{ owner, ownerVuln, advisory }]
const unresolved = new Set();
for (const [name, vuln] of Object.entries(vulns)) {
  if (vuln.severity !== 'high' && vuln.severity !== 'critical') continue;
  const r = resolve(name);
  for (const u of r.unresolved) unresolved.add(`${name} -> ${u}`);
  if (r.advisories.length === 0 && r.unresolved.length === 0) unresolved.add(`${name} (resolves to no advisory)`);
  for (const a of r.advisories) {
    if (!flagged.has(a.id)) flagged.set(a.id, []);
    flagged.get(a.id).push(a);
  }
}

if (unresolved.size > 0) {
  console.error(`::error::npm audit lists high/critical packages whose advisories cannot be resolved: ${[...unresolved].join(', ')}`);
  console.error('Failing closed: a transitive finding is never allowed by default.');
  process.exit(1);
}

if (flagged.size === 0) {
  if (npmAuditExitCode === 0) {
    console.log('npm audit: no high/critical advisories.');
    process.exit(0);
  }
  console.error(
    `npm audit exited ${npmAuditExitCode} (non-zero) but no high/critical advisory objects were found in its JSON.`
  );
  console.error('Failing closed rather than silently passing an unrecognized failure mode.');
  console.error(auditReportRaw.slice(0, 2000));
  process.exit(npmAuditExitCode || 1);
}

let allowlist = { exceptions: [] };
if (existsSync(allowlistPath)) {
  try {
    allowlist = JSON.parse(readFileSync(allowlistPath, 'utf-8'));
  } catch (err) {
    console.error(`${allowlistPath} exists but is not valid JSON: ${err.message}`);
    console.error('Fix the allowlist file — failing closed until it parses.');
    process.exit(1);
  }
}
if (!isObject(allowlist) || !Array.isArray(allowlist.exceptions ?? [])) {
  console.error(`${allowlistPath} must be an object whose \`exceptions\` is an array. Failing closed.`);
  process.exit(1);
}

const today = new Date().toISOString().slice(0, 10); // UTC YYYY-MM-DD
const validExceptions = new Map(); // id -> entry
const expiredExceptions = new Map(); // id -> expiresOn
for (const exception of allowlist.exceptions ?? []) {
  if (typeof exception?.id !== 'string' && typeof exception?.id !== 'number') continue;
  const fieldsOk =
    typeof exception.package === 'string' &&
    typeof exception.range === 'string' &&
    squash(exception.range) !== '' &&
    typeof exception.addedOn === 'string' &&
    DATE_RE.test(exception.addedOn) &&
    typeof exception.expiresOn === 'string' &&
    DATE_RE.test(exception.expiresOn);
  if (!fieldsOk) {
    console.error(`Exception ${exception.id} in ${allowlistPath} lacks package, range, addedOn or expiresOn (YYYY-MM-DD); it does not apply.`);
    continue;
  }
  if (exception.expiresOn >= today) {
    validExceptions.set(String(exception.id), exception);
  } else {
    expiredExceptions.set(String(exception.id), exception.expiresOn);
  }
}

const uncovered = [];
for (const [id, hits] of flagged) {
  const exception = validExceptions.get(id);
  const reasons = [];
  const notes = new Set();
  if (!exception) {
    reasons.push(
      expiredExceptions.has(id)
        ? `its exception expired on ${expiredExceptions.get(id)}`
        : 'no exception lists it'
    );
  } else {
    for (const { owner, ownerVuln, advisory } of hits) {
      if (exception.package !== owner || advisory.name !== owner) {
        reasons.push(`it is filed under ${owner}, but the exception names ${exception.package}`);
      }
      if (typeof advisory.range !== 'string' || squash(advisory.range) !== squash(exception.range)) {
        reasons.push(`its reported range is ${JSON.stringify(advisory.range)}, not the excepted ${exception.range}: re-check the advisory`);
      }
      const fix = fixFor(owner, ownerVuln.fixAvailable, notes);
      if (fix !== null) {
        reasons.push(`npm reports a fix for ${owner} (${fix}): bump it and remove the exception`);
      }
    }
  }
  const { owner, advisory } = hits[0];
  const label = `${id} (${ghsaOf(advisory)}) in ${owner} ${advisory.range ?? '(no range)'} [${advisory.severity ?? '?'}]`;
  if (reasons.length === 0) {
    console.log(
      `npm audit: advisory ${label} is covered by a reviewed exception in ${allowlistPath} (added ${exception.addedOn}, expires ${exception.expiresOn}).`
    );
    for (const note of notes) console.log(`npm audit:   ${note}.`);
  } else {
    uncovered.push(id);
    for (const reason of [...new Set(reasons)]) console.error(`npm audit: advisory ${label} is not covered: ${reason}.`);
  }
}

if (uncovered.length > 0) {
  console.error('');
  console.error(`::error::npm audit found high/critical advisories with no valid exception: ${uncovered.join(', ')}`);
  console.error(
    `Add a time-boxed entry to ${allowlistPath} (id, package, range, reason, approvedBy, addedOn, expiresOn — see`
  );
  console.error('its "description" field for the full procedure) via a normal reviewed PR, or fix/bump the');
  console.error('dependency instead. An expired exception is treated the same as no exception at all.');
  process.exit(1);
}

console.log('npm audit: all flagged advisories are covered by reviewed, non-expired exceptions — passing.');
process.exit(0);
