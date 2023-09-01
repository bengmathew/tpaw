/**
 * @generated SignedSource<<609849c67bab8ee5d43dc26fcc6b3cb1>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Query } from 'relay-runtime';
import { FragmentRefs } from "relay-runtime";
export type PlanRootServerQuery$variables = {
  includeUser: boolean;
  slug?: string | null;
  targetCount: number;
  userId: string;
};
export type PlanRootServerQuery$data = {
  readonly user?: {
    readonly plan: {
      readonly addedToServerAt: number;
      readonly id: string;
      readonly isMain: boolean;
      readonly label: string | null;
      readonly lastSyncAt: number;
      readonly planParamsPostBase: ReadonlyArray<{
        readonly change: string;
        readonly id: string;
        readonly params: string;
      }>;
      readonly reverseHeadIndex: number;
      readonly slug: string;
      readonly sortTime: number;
    };
  };
  readonly " $fragmentSpreads": FragmentRefs<"WithUser_query">;
};
export type PlanRootServerQuery = {
  response: PlanRootServerQuery$data;
  variables: PlanRootServerQuery$variables;
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
  "name": "slug"
},
v2 = {
  "defaultValue": null,
  "kind": "LocalArgument",
  "name": "targetCount"
},
v3 = {
  "defaultValue": null,
  "kind": "LocalArgument",
  "name": "userId"
},
v4 = [
  {
    "kind": "Variable",
    "name": "userId",
    "variableName": "userId"
  }
],
v5 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "id",
  "storageKey": null
},
v6 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "isMain",
  "storageKey": null
},
v7 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "label",
  "storageKey": null
},
v8 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "slug",
  "storageKey": null
},
v9 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "addedToServerAt",
  "storageKey": null
},
v10 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "sortTime",
  "storageKey": null
},
v11 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "lastSyncAt",
  "storageKey": null
},
v12 = {
  "alias": null,
  "args": [
    {
      "kind": "Variable",
      "name": "slug",
      "variableName": "slug"
    }
  ],
  "concreteType": "PlanWithHistory",
  "kind": "LinkedField",
  "name": "plan",
  "plural": false,
  "selections": [
    (v5/*: any*/),
    (v6/*: any*/),
    (v7/*: any*/),
    (v8/*: any*/),
    (v9/*: any*/),
    (v10/*: any*/),
    (v11/*: any*/),
    {
      "alias": null,
      "args": null,
      "kind": "ScalarField",
      "name": "reverseHeadIndex",
      "storageKey": null
    },
    {
      "alias": null,
      "args": [
        {
          "kind": "Variable",
          "name": "targetCount",
          "variableName": "targetCount"
        }
      ],
      "concreteType": "PlanParamsChangePatched",
      "kind": "LinkedField",
      "name": "planParamsPostBase",
      "plural": true,
      "selections": [
        (v5/*: any*/),
        {
          "alias": null,
          "args": null,
          "kind": "ScalarField",
          "name": "params",
          "storageKey": null
        },
        {
          "alias": null,
          "args": null,
          "kind": "ScalarField",
          "name": "change",
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
      (v3/*: any*/)
    ],
    "kind": "Fragment",
    "metadata": null,
    "name": "PlanRootServerQuery",
    "selections": [
      {
        "args": null,
        "kind": "FragmentSpread",
        "name": "WithUser_query"
      },
      {
        "condition": "includeUser",
        "kind": "Condition",
        "passingValue": true,
        "selections": [
          {
            "alias": null,
            "args": (v4/*: any*/),
            "concreteType": "User",
            "kind": "LinkedField",
            "name": "user",
            "plural": false,
            "selections": [
              (v12/*: any*/)
            ],
            "storageKey": null
          }
        ]
      }
    ],
    "type": "Query",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": [
      (v3/*: any*/),
      (v0/*: any*/),
      (v2/*: any*/),
      (v1/*: any*/)
    ],
    "kind": "Operation",
    "name": "PlanRootServerQuery",
    "selections": [
      {
        "condition": "includeUser",
        "kind": "Condition",
        "passingValue": true,
        "selections": [
          {
            "alias": null,
            "args": (v4/*: any*/),
            "concreteType": "User",
            "kind": "LinkedField",
            "name": "user",
            "plural": false,
            "selections": [
              (v5/*: any*/),
              {
                "alias": null,
                "args": null,
                "concreteType": "PlanWithHistory",
                "kind": "LinkedField",
                "name": "plans",
                "plural": true,
                "selections": [
                  (v5/*: any*/),
                  (v7/*: any*/),
                  (v8/*: any*/),
                  (v9/*: any*/),
                  (v10/*: any*/),
                  (v11/*: any*/),
                  (v6/*: any*/)
                ],
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "nonPlanParamsLastUpdatedAt",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "nonPlanParams",
                "storageKey": null
              },
              (v12/*: any*/)
            ],
            "storageKey": null
          }
        ]
      }
    ]
  },
  "params": {
    "cacheID": "b4ce6d902ccbf7bac897052466bedf4c",
    "id": null,
    "metadata": {},
    "name": "PlanRootServerQuery",
    "operationKind": "query",
    "text": "query PlanRootServerQuery(\n  $userId: ID!\n  $includeUser: Boolean!\n  $targetCount: Int!\n  $slug: String\n) {\n  ...WithUser_query\n  user(userId: $userId) @include(if: $includeUser) {\n    plan(slug: $slug) {\n      id\n      isMain\n      label\n      slug\n      addedToServerAt\n      sortTime\n      lastSyncAt\n      reverseHeadIndex\n      planParamsPostBase(targetCount: $targetCount) {\n        id\n        params\n        change\n      }\n    }\n    id\n  }\n}\n\nfragment WithUser_query on Query {\n  user(userId: $userId) @include(if: $includeUser) {\n    ...WithUser_user\n    id\n  }\n}\n\nfragment WithUser_user on User {\n  id\n  plans {\n    id\n    label\n    slug\n    addedToServerAt\n    sortTime\n    lastSyncAt\n    isMain\n  }\n  nonPlanParamsLastUpdatedAt\n  nonPlanParams\n}\n"
  }
};
})();

(node as any).hash = "ca43ae71e0699d6888534fffdd901ae5";

export default node;