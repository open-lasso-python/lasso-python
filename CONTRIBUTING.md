# Open LASSO Python Contribution Guide

## Roles

There are roughly two roles in which you can contribute:

- Contributors who just want to add changes from time to time
- Maintainers who oversee the repo, code themselves and review contribution
  before they can be merged

## Community

The community can be found on [discord].
Nothing beats a good discussion about existing features, new features or ideas
so jump right in.

[discord]: https://discord.gg/jYUgTsEWtN

## Spirit

We are all learners, some in the early stage some in the later.
In a code review, we take the patience to imaginarily sit down together and
explain other people why something is recommended differently or how things are
usually done in software or python.
This often seems tedious at first but growing together is important for any kind
of project which wants to grow itself.
So no fear in case of lack of experience but bring your learning spirit.
Samewise to any experienced developer, have patience and explain things.
Take the opportunity to sit down together on discord.

## How to make a Contribution

Tl;dr;

1. Fork the open lasso python repository
2. Clone the repo to your filesystem
3. Install [task][task_install]
4. Check out the `develop` branch
5. Set up the repo for development through `task setup`
6. Create a new branch from `develop` with the naming pattern `feature/...`
7. Make changes, commit and push them
8. Create a Pull Request in your for repo with target on the original repo
9. Add as reviewer `open-lasso-python/developers`
10. Wait for review patiently but you may nudge us a bit 🫶
11. Perform a Squash Merge and give a reasonable commit message as
    `<branch type>: <description>` where `branch_type` is one of the categories
    below.

[task_install]:https://taskfile.dev/installation/

You can fork the repo (1) by clicking on for in the top-right of the original
repo.
Cloning the repo (2) is traditionally done with git then of course.
Task is required (3) since it is used to store complex commands such as testing,
linting, build docs, etc.
(4) All development activities originate from the `develop` branch in which all
Pull Requests are finally merged again.
To create a branch choose a respective naming pattern following the angular
scheme: `<branch type>/<issue nr if exists>-<rough description/name>`.
Typical branch types are:

- `feature` for new features or if you got no clue what it is
- `bugfix` for 🐛-fixes
- `ci` for changes on the Continuous Integration pipeline
- `docs` for documentation related works
- `refactor` if the PR just does cleanup 🧹 and improves the code
- `test` for solely test-related work

Don't take these too seriously but they ought to provide rough categories.
**They are also used in the commit message when you squash merge a PR where it
is important!**
Install all dependencies otherwise obviously you can't code (5).
After making changes and pushing your branch to your forked repo (7 & 8), open a
Pull Request but make the target not `develop` in your fork but `develop` in the
original repo (7).
If not done automatically, add the maintainer group as reviewers (9).
Enjoy a healthy code review but be a bit patient with time as people contribute
voluntarily and may privately be occupied (10).
After approval, perform a Squash Merge (11).
A Squash Merge tosses away all the little, dirty commits we all do during
development.
What stays is the **final merge commit message and please pay attention here**
to format it right.
Why is this important?
This is needed to automatically generate a reasonable changelog during releases.
Thanks for contributing at this point.
Go wild and have fun 🥳
