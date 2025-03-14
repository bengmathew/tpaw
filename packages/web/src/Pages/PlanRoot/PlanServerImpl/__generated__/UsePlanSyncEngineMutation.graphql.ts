/**
 * @generated SignedSource<<2a6efded08610187206d20e9b33ec64d>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest } from 'relay-runtime';
import { FragmentRefs } from "relay-runtime";
export type UserPlanSyncInput = {
  add: ReadonlyArray<UserPlanSyncAddInput>;
  cutAfterId: string;
  lastSyncAt: number;
  planId: string;
  reverseHeadIndex: number;
  userId: string;
};
export type UserPlanSyncAddInput = {
  change: string;
  id: string;
  params: string;
};
export type UsePlanSyncEngineMutation$variables = {
  input: UserPlanSyncInput;
};
export type UsePlanSyncEngineMutation$data = {
  readonly userPlanSync: {
    readonly __typename: "ConcurrentChangeError";
    readonly _: number;
  } | {
    readonly __typename: "PlanAndUserResult";
    readonly plan: {
      readonly id: string;
      readonly lastSyncAt: number;
      readonly " $fragmentSpreads": FragmentRefs<"PlanWithoutParamsFragment">;
    };
  } | {
    // This will never be '%other', but we need some
    // value in case none of the concrete values match.
    readonly __typename: "%other";
  };
};
export type UsePlanSyncEngineMutation = {
  response: UsePlanSyncEngineMutation$data;
  variables: UsePlanSyncEngineMutation$variables;
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
    "kind": "Variable",
    "name": "input",
    "variableName": "input"
  }
],
v2 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "__typename",
  "storageKey": null
},
v3 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "id",
  "storageKey": null
},
v4 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "lastSyncAt",
  "storageKey": null
},
v5 = {
  "kind": "InlineFragment",
  "selections": [
    {
      "alias": null,
      "args": null,
      "kind": "ScalarField",
      "name": "_",
      "storageKey": null
    }
  ],
  "type": "ConcurrentChangeError",
  "abstractKey": null
};
return {
  "fragment": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Fragment",
    "metadata": null,
    "name": "UsePlanSyncEngineMutation",
    "selections": [
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": null,
        "kind": "LinkedField",
        "name": "userPlanSync",
        "plural": false,
        "selections": [
          (v2/*: any*/),
          {
            "kind": "InlineFragment",
            "selections": [
              {
                "alias": null,
                "args": null,
                "concreteType": "PlanWithHistory",
                "kind": "LinkedField",
                "name": "plan",
                "plural": false,
                "selections": [
                  (v3/*: any*/),
                  (v4/*: any*/),
                  {
                    "args": null,
                    "kind": "FragmentSpread",
                    "name": "PlanWithoutParamsFragment"
                  }
                ],
                "storageKey": null
              }
            ],
            "type": "PlanAndUserResult",
            "abstractKey": null
          },
          (v5/*: any*/)
        ],
        "storageKey": null
      }
    ],
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "UsePlanSyncEngineMutation",
    "selections": [
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": null,
        "kind": "LinkedField",
        "name": "userPlanSync",
        "plural": false,
        "selections": [
          (v2/*: any*/),
          {
            "kind": "InlineFragment",
            "selections": [
              {
                "alias": null,
                "args": null,
                "concreteType": "PlanWithHistory",
                "kind": "LinkedField",
                "name": "plan",
                "plural": false,
                "selections": [
                  (v3/*: any*/),
                  (v4/*: any*/),
                  {
                    "alias": null,
                    "args": null,
                    "kind": "ScalarField",
                    "name": "isMain",
                    "storageKey": null
                  },
                  {
                    "alias": null,
                    "args": null,
                    "kind": "ScalarField",
                    "name": "label",
                    "storageKey": null
                  },
                  {
                    "alias": null,
                    "args": null,
                    "kind": "ScalarField",
                    "name": "slug",
                    "storageKey": null
                  },
                  {
                    "alias": null,
                    "args": null,
                    "kind": "ScalarField",
                    "name": "addedToServerAt",
                    "storageKey": null
                  },
                  {
                    "alias": null,
                    "args": null,
                    "kind": "ScalarField",
                    "name": "sortTime",
                    "storageKey": null
                  }
                ],
                "storageKey": null
              }
            ],
            "type": "PlanAndUserResult",
            "abstractKey": null
          },
          (v5/*: any*/)
        ],
        "storageKey": null
      }
    ]
  },
  "params": {
    "cacheID": "3efa3776ce4ace782992880a0d8a5dd9",
    "id": null,
    "metadata": {},
    "name": "UsePlanSyncEngineMutation",
    "operationKind": "mutation",
    "text": "mutation UsePlanSyncEngineMutation(\n  $input: UserPlanSyncInput!\n) {\n  userPlanSync(input: $input) {\n    __typename\n    ... on PlanAndUserResult {\n      plan {\n        id\n        lastSyncAt\n        ...PlanWithoutParamsFragment\n      }\n    }\n    ... on ConcurrentChangeError {\n      _\n    }\n  }\n}\n\nfragment PlanWithoutParamsFragment on PlanWithHistory {\n  id\n  isMain\n  label\n  slug\n  addedToServerAt\n  sortTime\n  lastSyncAt\n}\n"
  }
};
})();

(node as any).hash = "0714a26feff8ca5d277e0e5b4325a16d";

export default node;
