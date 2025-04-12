module.exports = {
    extends: ["@commitlint/config-conventional"],
    rules: {
      "header-max-length": [2, "always", 72],
      "type-enum": [
        2,
        "always",
        [
          "build",
          "change",
          "chore",
          "ci",
          "disable",
          "docs",
          "feat",
          "fix",
          "perf",
          "refactor",
          "remove",
          "rename",
          "revert",
          "style",
          "test",
        ],
      ],
    },
  };
