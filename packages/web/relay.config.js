module.exports = {
  // ...
  src: "./src",
  language: "typescript", // "javascript" | "typescript" | "flow"
  schema: "../server/generated/schema.graphql",
  exclude: ["**/node_modules/**", "**/__mocks__/**", "**/__generated__/**"],
  noFutureProofEnums:true
}