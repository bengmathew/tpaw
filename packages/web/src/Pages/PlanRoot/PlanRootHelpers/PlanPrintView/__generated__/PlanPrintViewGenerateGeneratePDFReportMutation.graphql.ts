/**
 * @generated SignedSource<<7773e25360e6ad737f742158376f0b47>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest } from 'relay-runtime'
export type GeneratePDFReportInput = {
  auth?: string | null | undefined;
  devicePixelRatio: number;
  url: string;
  viewportHeight: number;
  viewportWidth: number;
};
export type PlanPrintViewGenerateGeneratePDFReportMutation$variables = {
  input: GeneratePDFReportInput;
};
export type PlanPrintViewGenerateGeneratePDFReportMutation$data = {
  readonly generatePDFReport: {
    readonly pdfURL: string;
  };
};
export type PlanPrintViewGenerateGeneratePDFReportMutation = {
  response: PlanPrintViewGenerateGeneratePDFReportMutation$data;
  variables: PlanPrintViewGenerateGeneratePDFReportMutation$variables;
};

const node: ConcreteRequest = (function(){
var v0 = [
  {
    "defaultValue": null,
    "kind": "LocalArgument",
    "name": "input"
  }
],
v1 = [
  {
    "alias": null,
    "args": [
      {
        "kind": "Variable",
        "name": "input",
        "variableName": "input"
      }
    ],
    "concreteType": "GeneratePDFReportOutput",
    "kind": "LinkedField",
    "name": "generatePDFReport",
    "plural": false,
    "selections": [
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "pdfURL",
        "storageKey": null
      }
    ],
    "storageKey": null
  }
];
return {
  "fragment": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Fragment",
    "metadata": null,
    "name": "PlanPrintViewGenerateGeneratePDFReportMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "PlanPrintViewGenerateGeneratePDFReportMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "92b004c950cba5f36dbccf3e69eb4076",
    "id": null,
    "metadata": {},
    "name": "PlanPrintViewGenerateGeneratePDFReportMutation",
    "operationKind": "mutation",
    "text": "mutation PlanPrintViewGenerateGeneratePDFReportMutation(\n  $input: GeneratePDFReportInput!\n) {\n  generatePDFReport(input: $input) {\n    pdfURL\n  }\n}\n"
  }
};
})();

(node as any).hash = "c454cda951bfcf7a1909d0d3b5227f2d";

export default node;
