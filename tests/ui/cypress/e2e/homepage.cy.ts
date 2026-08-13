describe('Check if text exists on the page', () => {
  it('should find the text on the page', () => {
    cy.visit('http://localhost:3000');
    cy.contains('GuideLLM').should('exist');
  });
});
