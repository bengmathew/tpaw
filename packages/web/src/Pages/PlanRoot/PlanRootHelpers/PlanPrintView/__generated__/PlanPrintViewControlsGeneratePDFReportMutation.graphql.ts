/**
 * @generated SignedSource<<c5c3a66f20434c87ec85e571a1ccaf1c>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Mutation } from 'relay-runtime';
export type GeneratePDFReportInput = {
  auth?: string | null | undefined;
  devicePixelRatio: number;
  url: string;
  viewportHeight: number;
  viewportWidth: number;
};
export type PlanPrintViewControlsGeneratePDFReportMutation$variables = {
  input: GeneratePDFReportInput;
};
export type PlanPrintViewControlsGeneratePDFReportMutation$data = {
  readonly generatePDFReport: {
    readonly pdfURL: string;
  };
};
export type PlanPrintViewControlsGeneratePDFReportMutation = {
  response: PlanPrintViewControlsGeneratePDFReportMutation$data;
  variables: PlanPrintViewControlsGeneratePDFReportMutation$variables;
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
    "name": "PlanPrintViewControlsGeneratePDFReportMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "PlanPrintViewControlsGeneratePDFReportMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "d941f17841960764e7c9cb3da64c4054",
    "id": null,
    "metadata": {},
    "name": "PlanPrintViewControlsGeneratePDFReportMutation",
    "operationKind": "mutation",
    "text": "mutation PlanPrintViewControlsGeneratePDFReportMutation(\n  $input: GeneratePDFReportInput!\n) {\n  generatePDFReport(input: $input) {\n    pdfURL\n  }\n}\n"
  }
};
})();

(node as any).hash = "8cfbe32c0b129f31cd6aa5baba9cd1aa";

export default node;
