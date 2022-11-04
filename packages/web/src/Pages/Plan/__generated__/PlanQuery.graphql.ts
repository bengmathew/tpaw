/**
 * @generated SignedSource<<cc1ee0ff32006950c68502131df91dbe>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Query } from 'relay-runtime';
import { FragmentRefs } from "relay-runtime";
export type PlanQuery$variables = {
  includeUser: boolean;
  userId: string;
};
export type PlanQuery$data = {
  readonly " $fragmentSpreads": FragmentRefs<"UserFragment_query">;
};
export type PlanQuery = {
  response: PlanQuery$data;
  variables: PlanQuery$variables;
};

const node: ConcreteRequest = (function(){
var v0 = {
  "defaultValue": null,
  "kind": "LocalArgument",
  "name": "includeUser"
},
v1 = {
  "defaultValue": null,
  "kind": "LocalArgument",
  "name": "userId"
},
v2 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "id",
  "storageKey": null
};
return {
  "fragment": {
    "argumentDefinitions": [
      (v0/*: any*/),
      (v1/*: any*/)
    ],
    "kind": "Fragment",
    "metadata": null,
    "name": "PlanQuery",
    "selections": [
      {
        "args": null,
        "kind": "FragmentSpread",
        "name": "UserFragment_query"
      }
    ],
    "type": "Query",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": [
      (v1/*: any*/),
      (v0/*: any*/)
    ],
    "kind": "Operation",
    "name": "PlanQuery",
    "selections": [
      {
        "condition": "includeUser",
        "kind": "Condition",
        "passingValue": true,
        "selections": [
          {
            "alias": null,
            "args": [
              {
                "kind": "Variable",
                "name": "userId",
                "variableName": "userId"
              }
            ],
            "concreteType": "User",
            "kind": "LinkedField",
            "name": "user",
            "plural": false,
            "selections": [
              (v2/*: any*/),
              {
                "alias": null,
                "args": null,
                "concreteType": "Plan",
                "kind": "LinkedField",
                "name": "plan",
                "plural": false,
                "selections": [
                  (v2/*: any*/),
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
                    "name": "modifiedAt",
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
            ],
            "storageKey": null
          }
        ]
      }
    ]
  },
  "params": {
    "cacheID": "ca43e66471d086afc0cb0d6329501f70",
    "id": null,
    "metadata": {},
    "name": "PlanQuery",
    "operationKind": "query",
    "text": "query PlanQuery(\n  $userId: ID!\n  $includeUser: Boolean!\n) {\n  ...UserFragment_query\n}\n\nfragment UserFragment_query on Query {\n  user(userId: $userId) @include(if: $includeUser) {\n    ...UserFragment_user\n    id\n  }\n}\n\nfragment UserFragment_user on User {\n  id\n  plan {\n    id\n    createdAt\n    modifiedAt\n    params\n  }\n}\n"
  }
};
})();

(node as any).hash = "720dc98f0b97aca1b45df6260f721241";

export default node;
