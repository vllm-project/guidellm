// @ts-check

import eslint from '@eslint/js';
import nextPlugin from '@next/eslint-plugin-next';
import prettierConfig from 'eslint-config-prettier';
import cypressPlugin from 'eslint-plugin-cypress';
import importPlugin from 'eslint-plugin-import';
import jestPlugin from 'eslint-plugin-jest';
import prettierPlugin from 'eslint-plugin-prettier';
import reactPlugin from 'eslint-plugin-react';
import hooksPlugin from 'eslint-plugin-react-hooks';
import globals from 'globals';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import tseslint from 'typescript-eslint';

// --- SETUP ---
// Recreate __dirname for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --- EXPORT ESLINT CONFIG ---
export default tseslint.config(
  // 1. Global Ignores
  {
    ignores: ['node_modules/', '.next/', 'dist/', 'coverage/', '.DS_Store'],
  },

  // 2. Base Configurations (Applied to all files)
  eslint.configs.recommended,
  prettierConfig, // Disables ESLint rules that conflict with Prettier. IMPORTANT: Must be after other configs.

  // 3. Configuration for App Source Code (Next.js with Type-Aware Linting)
  {
    files: ['src/ui/**/*.{ts,tsx}'],
    languageOptions: {
      parser: tseslint.parser,
      parserOptions: {
        project: true, // Enable type-aware linting
        tsconfigRootDir: __dirname,
      },
      globals: {
        ...globals.browser,
        ...globals.node, // Add Node.js globals for `process` etc.
      },
    },
    plugins: {
      '@typescript-eslint': tseslint.plugin,
      '@next/next': nextPlugin,
      import: importPlugin,
      react: reactPlugin,
      'react-hooks': hooksPlugin,
      prettier: prettierPlugin,
    },
    rules: {
      // --- Base rules to disable in favor of TS versions ---
      'no-unused-vars': 'off',

      // --- Recommended rules from plugins ---
      ...tseslint.configs.recommendedTypeChecked.rules,
      ...nextPlugin.configs.recommended.rules,
      ...nextPlugin.configs['core-web-vitals'].rules,
      ...reactPlugin.configs.recommended.rules,
      ...hooksPlugin.configs.recommended.rules,

      // --- Prettier ---
      'prettier/prettier': 'error',

      // --- Custom Rules & Overrides ---
      '@typescript-eslint/no-unused-vars': [
        'warn',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
      '@typescript-eslint/no-floating-promises': 'error',
      '@typescript-eslint/no-misused-promises': 'error',
      '@typescript-eslint/no-explicit-any': 'warn',

      'import/order': [
        'error',
        {
          groups: [['builtin', 'external'], 'internal', ['parent', 'sibling', 'index']],
          'newlines-between': 'always',
          alphabetize: { order: 'asc', caseInsensitive: true },
        },
      ],

      'react/react-in-jsx-scope': 'off',
      'react/prop-types': 'off',

      '@next/next/no-html-link-for-pages': 'off',
      '@next/next/no-img-element': 'off',

      complexity: ['warn', { max: 8 }],
    },
    settings: {
      react: { version: 'detect' },
      'import/resolver': { typescript: true, node: true },
    },
  },

  // 4. Configuration for Jest Test Files (Type-Aware)
  {
    files: ['tests/ui/**/*.{test,spec}.{ts,tsx}', 'jest.setup.ts'],
    languageOptions: {
      parser: tseslint.parser, // Explicitly set parser
      parserOptions: {
        project: './tsconfig.test.json',
        tsconfigRootDir: __dirname,
      },
      globals: {
        ...globals.jest,
        ...globals.node, // FIX: Add Node.js globals for `global`, etc.
      },
    },
    plugins: {
      jest: jestPlugin,
    },
    rules: {
      ...jestPlugin.configs['flat/recommended'].rules,
      '@typescript-eslint/unbound-method': 'off',
    },
  },

  // 5. Configuration for Cypress E2E Test Files (Type-Aware)
  {
    files: [
      'tests/ui/cypress/**/*.{cy,e2e}.{ts,tsx}',
      'tests/ui/cypress/support/**/*.ts',
    ],
    languageOptions: {
      parser: tseslint.parser, // Explicitly set parser
      parserOptions: {
        project: './tsconfig.cypress.json',
        tsconfigRootDir: __dirname,
      },
    },
    plugins: {
      cypress: cypressPlugin,
    },
    // Apply recommended rules and then add our overrides
    rules: {
      ...cypressPlugin.configs.recommended.rules,
      'jest/expect-expect': 'off',
      'jest/no-standalone-expect': 'off',
      '@typescript-eslint/no-floating-promises': 'off',
    },
  },

  // 6. Configuration for JS/TS config files
  {
    files: ['**/*.config.{js,mjs,ts}'],
    languageOptions: {
      globals: {
        ...globals.node,
      },
    },
    rules: {
      '@typescript-eslint/no-var-requires': 'off',
    },
  },

  // 7. Configuration for JS/TS mock files and test helpers
  {
    files: ['tests/ui/**/__mocks__/**/*.{js,ts}', 'tests/ui/unit/mocks/**/*.ts'],
    languageOptions: {
      globals: {
        ...globals.node,
      },
    },
  }
);
