/**
 * @generated SignedSource<<e62e7332d119e804f35fa4ffa4800a3b>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, FragmentRefs } from 'relay-runtime'
export type UserPlanResetInput = {
  ianaTimezoneName: string;
  lastSyncAt: number;
  planId: string;
  userId: string;
};
export type PlanMenuActionModalResetMutation$variables = {
  input: UserPlanResetInput;
};
export type PlanMenuActionModalResetMutation$data = {
  readonly userPlanReset: {
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
export type PlanMenuActionModalResetMutation = {
  response: PlanMenuActionModalResetMutation$data;
  variables: PlanMenuActionModalResetMutation$variables;
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
    "name": "PlanMenuActionModalResetMutation",
    "selections": [
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": null,
        "kind": "LinkedField",
        "name": "userPlanReset",
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
    "name": "PlanMenuActionModalResetMutation",
    "selections": [
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": null,
        "kind": "LinkedField",
        "name": "userPlanReset",
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
    "cacheID": "71568ba2d3126ed2d5864f2d031990b2",
    "id": null,
    "metadata": {},
    "name": "PlanMenuActionModalResetMutation",
    "operationKind": "mutation",
    "text": "mutation PlanMenuActionModalResetMutation(\n  $input: UserPlanResetInput!\n) {\n  userPlanReset(input: $input) {\n    __typename\n    ... on PlanAndUserResult {\n      plan {\n        id\n        lastSyncAt\n        ...PlanWithoutParamsFragment\n      }\n    }\n    ... on ConcurrentChangeError {\n      _\n    }\n  }\n}\n\nfragment PlanWithoutParamsFragment on PlanWithHistory {\n  id\n  isMain\n  label\n  slug\n  addedToServerAt\n  sortTime\n  lastSyncAt\n}\n"
  }
};
})();

(node as any).hash = "f000ad80d445293a42694fd46f856ecf";

export default node;
