/**
 * @generated SignedSource<<2235f08e55c72265cb3c5270c0abaab0>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Query } from 'relay-runtime';
export type WithURLPlanParamsGetParamsQuery$variables = {
  linkId: string;
};
export type WithURLPlanParamsGetParamsQuery$data = {
  readonly linkBasedPlan: {
    readonly createdAt: number;
    readonly id: string;
    readonly params: string;
  } | null;
};
export type WithURLPlanParamsGetParamsQuery = {
  response: WithURLPlanParamsGetParamsQuery$data;
  variables: WithURLPlanParamsGetParamsQuery$variables;
};

const node: ConcreteRequest = (function(){
var v0 = [
  {
    "defaultValue": null,
    "kind": "LocalArgument",
    "name": "linkId"
  }
],
v1 = [
  {
    "alias": null,
    "args": [
      {
        "kind": "Variable",
        "name": "linkId",
        "variableName": "linkId"
      }
    ],
    "concreteType": "LinkBasedPlan",
    "kind": "LinkedField",
    "name": "linkBasedPlan",
    "plural": false,
    "selections": [
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "id",
        "storageKey": null
      },
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "createdAt",
        "storageKey": null
      },
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "params",
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
    "name": "WithURLPlanParamsGetParamsQuery",
    "selections": (v1/*: any*/),
    "type": "Query",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "WithURLPlanParamsGetParamsQuery",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "fbf071aeceeefbbe82843fd01400b1c8",
    "id": null,
    "metadata": {},
    "name": "WithURLPlanParamsGetParamsQuery",
    "operationKind": "query",
    "text": "query WithURLPlanParamsGetParamsQuery(\n  $linkId: ID!\n) {\n  linkBasedPlan(linkId: $linkId) {\n    id\n    createdAt\n    params\n  }\n}\n"
  }
};
})();

(node as any).hash = "9c9a77102a1e7bfeacc6cdc1327b1ccd";

export default node;
