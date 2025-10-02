# Branch Protection Setup Guide

## Recommended Branch Protection Rules for `main`

To set up proper branch protection for the python-optimizer repository, go to:

**GitHub Repository → Settings → Branches → Add rule**

### Branch Name Pattern
```
main
```

### Protection Rules to Enable:

#### ✅ **Require a pull request before merging**
- [x] Require approvals: **1**
- [x] Dismiss stale PR approvals when new commits are pushed
- [x] Require review from code owners (if CODEOWNERS file exists)

#### ✅ **Require status checks to pass before merging**
- [x] Require branches to be up to date before merging
- **Required status checks:**
  - `Test Suite (ubuntu-latest, Python 3.11)` - Main test environment
  - `Test Suite (ubuntu-latest, Python 3.12)` - Latest Python version
  - `Performance Benchmarks` - Ensure no regressions
  - `Security Scan` - Security vulnerability checks

#### ✅ **Require conversation resolution before merging**
- [x] All conversations on code must be resolved

#### ✅ **Require signed commits**  
- [x] Require signed commits (recommended for security)

#### ✅ **Require linear history**
- [x] Require linear history (keeps clean git history)

#### ✅ **Do not allow bypassing the above settings**
- [x] Include administrators (ensures rules apply to all)

#### ✅ **Restrict pushes that create files**
- [x] Restrict pushes that create files (prevents accidental large files)

### Optional Advanced Settings:

#### **Auto-merge settings:**
- [x] Allow auto-merge (after all checks pass)
- [x] Automatically delete head branches (clean up after merge)

#### **Required deployments:**
- If using deployment environments, require successful deployment to staging

## Workflow Integration

### GitHub Actions Status Checks
The following GitHub Actions jobs should be required:
```yaml
# From .github/workflows/ci.yml
- Test Suite (ubuntu-latest, Python 3.11)
- Test Suite (ubuntu-latest, Python 3.12)  
- Performance Benchmarks
- Security Scan
```

### Development Workflow
1. **Create feature branch** from `main`
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and commit**
   ```bash
   git add .
   git commit -m "feat: description of changes"
   ```

3. **Push to feature branch**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create Pull Request** in GitHub
   - Base: `main`
   - Compare: `feature/your-feature-name`
   - Add description and link any issues

5. **Wait for checks** and review approval

6. **Merge** (squash and merge recommended)

## CODEOWNERS File (Optional)

Create `.github/CODEOWNERS` to require specific reviewers:

```
# Global owners
* @thinmanj

# Python code
*.py @thinmanj

# CI/CD configurations  
.github/ @thinmanj
Makefile @thinmanj
pyproject.toml @thinmanj

# Documentation
docs/ @thinmanj
README.md @thinmanj
```

## Emergency Bypass

In case of emergencies, repository administrators can:
1. Temporarily disable branch protection
2. Push critical fixes
3. Re-enable protection immediately

## Benefits of This Setup

✅ **Code Quality** - All code reviewed before merging
✅ **CI/CD Integration** - Tests must pass before merge  
✅ **Security** - Signed commits and security scans
✅ **Clean History** - Linear history and no direct pushes to main
✅ **Documentation** - PR descriptions document changes
✅ **Collaboration** - Review process ensures knowledge sharing

## Current Branch Strategy

- **`main`** - Production-ready code only
- **`feature/`** - Feature development branches
- **`hotfix/`** - Critical bug fixes
- **`release/`** - Release preparation branches