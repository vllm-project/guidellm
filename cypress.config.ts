import { defineConfig } from 'cypress';

export default defineConfig({
  e2e: {
    specPattern: 'tests/ui/cypress/e2e/**/*.cy.{js,jsx,ts,tsx}',
    supportFile: 'tests/ui/cypress/support/e2e.ts',
    baseUrl: 'http://localhost:3000', // optional
  },
});
