module.exports = {
  extends: 'next/core-web-vitals',
  rules: {
    // Generally applicable.
    '@typescript-eslint/no-floating-promises': 'error',
    '@typescript-eslint/no-empty-function': 'off',
    '@typescript-eslint/no-namespace': 'off',
    '@typescript-eslint/require-await': 'off',

    // Project specific.
    indent: 'off',
    'react/display-name': 'off',
    'react-hooks/rules-of-hooks': 'error',
    'react-hooks/exhaustive-deps': 'error',
  },
}
