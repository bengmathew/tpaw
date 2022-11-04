module.exports = {
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 2018,
    project: 'tsconfig.json',
    tsconfigRootDir: __dirname,
    sourceType: 'module',
  },
  extends: [
    'plugin:@typescript-eslint/recommended',
    'plugin:@typescript-eslint/recommended-requiring-type-checking',
    'plugin:promise/recommended',
  ],
  plugins: ['promise', '@typescript-eslint/eslint-plugin'],
  root: true,
  env: {
    node: true,
  },
  ignorePatterns: ['.eslintrc.js', 'generated/**'],
  rules: {
    // Generally applicable.
    '@typescript-eslint/no-floating-promises': 'error',
    '@typescript-eslint/no-empty-function': 'off',
    '@typescript-eslint/no-namespace': 'off',
    '@typescript-eslint/require-await': 'off',

    // Project specific.
  },
}
