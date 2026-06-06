# Branch protection for `master`

Enable this in GitHub: Settings -> Branches -> Add branch protection rule.

Recommended rule:

- Branch name pattern: `master`
- Require status checks to pass before merging: enabled
- Required status check: `Test`
- Require branches to be up to date before merging: enabled
- Require a pull request before merging: optional for solo work, recommended
- Do not allow bypassing the above settings: enabled if available
- Restrict who can push to matching branches: enabled if you want `master` to
  be deploy-only

Production deploy is handled by the server-side `rqm-auto-deploy.timer`, which
polls `origin/master` and runs `scripts/update_server.sh` when a new commit is
available.
