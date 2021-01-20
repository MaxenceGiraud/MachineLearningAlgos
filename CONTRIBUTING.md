# Contributions

This repo is only made by myself ([Maxence Giraud](https://github.com/MaxenceGiraud/)), I do not seek additional contributors as it has only learning purposes and no other intentions.
You can still open an issue if you find a mistake or have a constructive comment.

## Git Commit Conventions
* Use the present tense ("Add feature" instead of "Adding feature" or "Added feature").
* Use the imperative mood ("Remove attribute" not "Removes attribute").
* Limit the first line to 72 characters or less.
* More details can be specified in the subsequent lines, *i.e.* in the body of the commit.
* Commits must be named as follows: `[Type] Description`.

### Commit Types
Consider starting the commit message with the following "types" (should be placed in brackets):
* **[_example_]**: 
* **Feat**: Adding a new (important) feature
* **Fix**: a bug fix or fixing test
* **Test**: adding of new tests
* **Clean**: (re)move some files / folders
* **Refactor**: refactoring code
* **Merge**: merging commits / branches
* **Doc**: changes to the documentation

A second type can be added if working on a specific sub project (ex: **[__DL__]** for deep learning).

### Examples
* Simple commit without body
```
[Add] Add the possibility to create a puzzle from an image
```
* More details provided in the body of the commit
```git
[Feat] Implement new ML technics
* Add new function `convolution`
* Moved cdist function to 'doc' folder
```
* Reference and close an issue reported on Github
```git
[Fix] Fix infinite loop in SMO SVD solver  #5
```
