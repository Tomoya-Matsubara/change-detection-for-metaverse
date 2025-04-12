module.exports = {
    disableEmoji: true,
    format: "{type}{scope}: {subject}",
    list: [
      "change",
      "chore",
      "ci",
      "disable",
      "docs",
      "feat",
      "fix",
      "perf",
      "refactor",
      "release",
      "remove",
      "rename",
      "style",
      "test",
    ],
    maxMessageLength: 72,
    minMessageLength: 3,
    questions: ["type", "scope", "subject", "body", "breaking", "issues"],
    scopes: ["", "git"],
    types: {
      change: {
        description: "Changes according to the specifications update",
        value: "change",
      },
      chore: {
        description: "Build process or auxiliary tool changes",
        value: "chore",
      },
      ci: {
        description: "CI related changes",
        value: "ci",
      },
      disable: {
        description: "Disable a feature",
        value: "disable",
      },
      docs: {
        description: "Documentation only changes",
        value: "docs",
      },
      feat: {
        description: "A new feature",
        value: "feat",
      },
      fix: {
        description: "A bug fix",
        value: "fix",
      },
      perf: {
        description: "A code change that improves performance",
        value: "perf",
      },
      refactor: {
        description: "A code change that neither fixes a bug or adds a feature",
        value: "refactor",
      },
      release: {
        description: "Create a release commit",
        value: "release",
      },
      remove: {
        description: "Remove features or files",
        value: "remove",
      },
      rename: {
        description: "Rename files",
        value: "rename",
      },
      style: {
        description: "Markup, white-space, formatting, missing semi-colons...",
        value: "style",
      },
      test: {
        description: "Add missing tests",
        value: "test",
      },
      messages: {
        type: "Select the type of change that you're committing:",
        customScope: "Select the scope this component affects:",
        subject: "Write a short, imperative mood description of the change:\n",
        body: "Provide a longer description of the change:\n ",
        breaking: "List any breaking changes:\n",
        footer: "Issues this commit closes, e.g #123:",
        confirmCommit: "The packages that this commit has affected\n",
      },
    },
  };
