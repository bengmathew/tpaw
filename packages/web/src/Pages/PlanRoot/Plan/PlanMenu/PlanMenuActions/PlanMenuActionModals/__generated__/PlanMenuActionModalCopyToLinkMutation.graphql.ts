/**
 * @generated SignedSource<<06bdda5e3befc88ed17cf9722bd601fd>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest } from 'relay-runtime';
export type CreateLinkBasedPlanInput = {
  params: string;
};
export type PlanMenuActionModalCopyToLinkMutation$variables = {
  input: CreateLinkBasedPlanInput;
};
export type PlanMenuActionModalCopyToLinkMutation$data = {
  readonly createLinkBasedPlan: {
    readonly id: string;
  };
};
export type PlanMenuActionModalCopyToLinkMutation = {
  response: PlanMenuActionModalCopyToLinkMutation$data;
  variables: PlanMenuActionModalCopyToLinkMutation$variables;
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
    "concreteType": "LinkBasedPlan",
    "kind": "LinkedField",
    "name": "createLinkBasedPlan",
    "plural": false,
    "selections": [
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "id",
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
    "name": "PlanMenuActionModalCopyToLinkMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "PlanMenuActionModalCopyToLinkMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "dfdea94e8c8aec0045a1c577754dd6e9",
    "id": null,
    "metadata": {},
    "name": "PlanMenuActionModalCopyToLinkMutation",
    "operationKind": "mutation",
    "text": "mutation PlanMenuActionModalCopyToLinkMutation(\n  $input: CreateLinkBasedPlanInput!\n) {\n  createLinkBasedPlan(input: $input) {\n    id\n  }\n}\n"
  }
};
})();

(node as any).hash = "291c0450cb3012b2d1626fa105a76815";

export default node;
