/**
 * @generated SignedSource<<2a37f591b3b2081c9693b45f8698412b>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Query } from 'relay-runtime';
export type UseServerHistoryFromStartQuery$variables = {
  baseId: string;
  baseTimestamp: number;
  ianaTimezoneName: string;
  planId: string;
  userId: string;
};
export type UseServerHistoryFromStartQuery$data = {
  readonly user: {
    readonly plan: {
      readonly id: string;
      readonly planParamsPreBase: ReadonlyArray<{
        readonly id: string;
        readonly params: string;
      }>;
    };
  };
};
export type UseServerHistoryFromStartQuery = {
  response: UseServerHistoryFromStartQuery$data;
  variables: UseServerHistoryFromStartQuery$variables;
};

const node: ConcreteRequest = (function(){
var v0 = {
  "defaultValue": null,
  "kind": "LocalArgument",
  "name": "baseId"
},
v1 = {
  "defaultValue": null,
  "kind": "LocalArgument",
  "name": "baseTimestamp"
},
v2 = {
  "defaultValue": null,
  "kind": "LocalArgument",
  "name": "ianaTimezoneName"
},
v3 = {
  "defaultValue": null,
  "kind": "LocalArgument",
  "name": "planId"
},
v4 = {
  "defaultValue": null,
  "kind": "LocalArgument",
  "name": "userId"
},
v5 = [
  {
    "kind": "Variable",
    "name": "userId",
    "variableName": "userId"
  }
],
v6 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "id",
  "storageKey": null
},
v7 = {
  "alias": null,
  "args": [
    {
      "kind": "Variable",
      "name": "planId",
      "variableName": "planId"
    }
  ],
  "concreteType": "PlanWithHistory",
  "kind": "LinkedField",
  "name": "plan",
  "plural": false,
  "selections": [
    (v6/*: any*/),
    {
      "alias": null,
      "args": [
        {
          "kind": "Variable",
          "name": "baseId",
          "variableName": "baseId"
        },
        {
          "kind": "Variable",
          "name": "baseTimestamp",
          "variableName": "baseTimestamp"
        },
        {
          "kind": "Variable",
          "name": "ianaTimezoneName",
          "variableName": "ianaTimezoneName"
        }
      ],
      "concreteType": "PlanParamsChangePatched",
      "kind": "LinkedField",
      "name": "planParamsPreBase",
      "plural": true,
      "selections": [
        (v6/*: any*/),
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
  ],
  "storageKey": null
};
return {
  "fragment": {
    "argumentDefinitions": [
      (v0/*: any*/),
      (v1/*: any*/),
      (v2/*: any*/),
      (v3/*: any*/),
      (v4/*: any*/)
    ],
    "kind": "Fragment",
    "metadata": null,
    "name": "UseServerHistoryFromStartQuery",
    "selections": [
      {
        "alias": null,
        "args": (v5/*: any*/),
        "concreteType": "User",
        "kind": "LinkedField",
        "name": "user",
        "plural": false,
        "selections": [
          (v7/*: any*/)
        ],
        "storageKey": null
      }
    ],
    "type": "Query",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": [
      (v4/*: any*/),
      (v3/*: any*/),
      (v1/*: any*/),
      (v0/*: any*/),
      (v2/*: any*/)
    ],
    "kind": "Operation",
    "name": "UseServerHistoryFromStartQuery",
    "selections": [
      {
        "alias": null,
        "args": (v5/*: any*/),
        "concreteType": "User",
        "kind": "LinkedField",
        "name": "user",
        "plural": false,
        "selections": [
          (v7/*: any*/),
          (v6/*: any*/)
        ],
        "storageKey": null
      }
    ]
  },
  "params": {
    "cacheID": "8eefa90346d9508960428f688998e37c",
    "id": null,
    "metadata": {},
    "name": "UseServerHistoryFromStartQuery",
    "operationKind": "query",
    "text": "query UseServerHistoryFromStartQuery(\n  $userId: ID!\n  $planId: String!\n  $baseTimestamp: Float!\n  $baseId: String!\n  $ianaTimezoneName: String!\n) {\n  user(userId: $userId) {\n    plan(planId: $planId) {\n      id\n      planParamsPreBase(baseTimestamp: $baseTimestamp, baseId: $baseId, ianaTimezoneName: $ianaTimezoneName) {\n        id\n        params\n      }\n    }\n    id\n  }\n}\n"
  }
};
})();

(node as any).hash = "fddd77ee82c719fbc24ffaf4492568b8";

export default node;
