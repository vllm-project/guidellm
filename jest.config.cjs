const path = require('path');
const nextJest = require('next/jest');

const createJestConfig = nextJest({
  dir: path.resolve(__dirname, 'src/ui'),
});

const customJestConfig = {
  collectCoverage: false,
  collectCoverageFrom: ['./src/ui/**/*.{ts,tsx}'],
  coverageDirectory: './coverage',
  coverageProvider: 'v8',
  coverageReporters: ['text-summary', 'lcov', 'json-summary'],
  moduleFileExtensions: ['ts', 'tsx', 'js'],
  moduleNameMapper: {
    '^.+\\.(svg)$': '<rootDir>/tests/ui/__mocks__/svg.js',
    '^@/(.*)$': '<rootDir>/src/$1',
    '\\.(jpg|jpeg|png|gif|eot|otf|webp|svg|ttf|woff|woff2|mp4|webm|wav|mp3|m4a|aac|oga)$':
      '<rootDir>/tests/ui/__mocks__/fileMock.js',
    '\\.(css|less|scss|sass)$': '<rootDir>/tests/ui/__mocks__/styleMock.js',
  },
  setupFilesAfterEnv: ['<rootDir>/jest.setup.ts'],
  testEnvironment: 'jest-environment-jsdom',
  testMatch: [
    '<rootDir>/tests/ui/unit/**/*.(test|spec).{ts,tsx,js,jsx}',
    '<rootDir>/tests/ui/integration/**/*.(test|spec).{ts,tsx,js,jsx}',
  ],
};

module.exports = createJestConfig(customJestConfig);
