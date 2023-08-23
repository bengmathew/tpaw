/**
 * @generated SignedSource<<dc5e262f0f951fff42c444eed9c20548>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Mutation } from 'relay-runtime';
import { FragmentRefs } from "relay-runtime";
export type UserMergeFromClientInput = {
  guestPlan?: UserPlanCreatePlanInput | null;
  linkPlan?: UserMergeFromClientLinkPlanInput | null;
  nonPlanParams?: string | null;
  userId: string;
};
export type UserPlanCreatePlanInput = {
  planParamsHistory: ReadonlyArray<UserPlanCreatePlanParamsHistryInput>;
  reverseHeadIndex: number;
};
export type UserPlanCreatePlanParamsHistryInput = {
  change: string;
  id: string;
  params: string;
};
export type UserMergeFromClientLinkPlanInput = {
  label: string;
  plan: UserPlanCreatePlanInput;
};
export type WithMergeToServerMutation$variables = {
  input: UserMergeFromClientInput;
};
export type WithMergeToServerMutation$data = {
  readonly userMergeFromClient: {
    readonly guestPlan: {
      readonly label: string | null;
      readonly " $fragmentSpreads": FragmentRefs<"PlanWithoutParamsFragment">;
    } | null;
    readonly linkPlan: {
      readonly label: string | null;
      readonly " $fragmentSpreads": FragmentRefs<"PlanWithoutParamsFragment">;
    } | null;
  };
};
export type WithMergeToServerMutation = {
  response: WithMergeToServerMutation$data;
  variables: WithMergeToServerMutation$variables;
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
  "name": "label",
  "storageKey": null
},
v3 = [
  (v2/*: any*/),
  {
    "args": null,
    "kind": "FragmentSpread",
    "name": "PlanWithoutParamsFragment"
  }
],
v4 = [
  (v2/*: any*/),
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
    "name": "isMain",
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
  },
  {
    "alias": null,
    "args": null,
    "kind": "ScalarField",
    "name": "lastSyncAt",
    "storageKey": null
  }
];
return {
  "fragment": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Fragment",
    "metadata": null,
    "name": "WithMergeToServerMutation",
    "selections": [
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": "UserMergeFromClientResult",
        "kind": "LinkedField",
        "name": "userMergeFromClient",
        "plural": false,
        "selections": [
          {
            "alias": null,
            "args": null,
            "concreteType": "PlanWithHistory",
            "kind": "LinkedField",
            "name": "guestPlan",
            "plural": false,
            "selections": (v3/*: any*/),
            "storageKey": null
          },
          {
            "alias": null,
            "args": null,
            "concreteType": "PlanWithHistory",
            "kind": "LinkedField",
            "name": "linkPlan",
            "plural": false,
            "selections": (v3/*: any*/),
            "storageKey": null
          }
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
    "name": "WithMergeToServerMutation",
    "selections": [
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": "UserMergeFromClientResult",
        "kind": "LinkedField",
        "name": "userMergeFromClient",
        "plural": false,
        "selections": [
          {
            "alias": null,
            "args": null,
            "concreteType": "PlanWithHistory",
            "kind": "LinkedField",
            "name": "guestPlan",
            "plural": false,
            "selections": (v4/*: any*/),
            "storageKey": null
          },
          {
            "alias": null,
            "args": null,
            "concreteType": "PlanWithHistory",
            "kind": "LinkedField",
            "name": "linkPlan",
            "plural": false,
            "selections": (v4/*: any*/),
            "storageKey": null
          }
        ],
        "storageKey": null
      }
    ]
  },
  "params": {
    "cacheID": "309e001aa79c830f20f5910000646058",
    "id": null,
    "metadata": {},
    "name": "WithMergeToServerMutation",
    "operationKind": "mutation",
    "text": "mutation WithMergeToServerMutation(\n  $input: UserMergeFromClientInput!\n) {\n  userMergeFromClient(input: $input) {\n    guestPlan {\n      label\n      ...PlanWithoutParamsFragment\n      id\n    }\n    linkPlan {\n      label\n      ...PlanWithoutParamsFragment\n      id\n    }\n  }\n}\n\nfragment PlanWithoutParamsFragment on PlanWithHistory {\n  id\n  isMain\n  label\n  slug\n  addedToServerAt\n  sortTime\n  lastSyncAt\n}\n"
  }
};
})();

(node as any).hash = "8efefa29cbb8eeece5050a8bd2143b15";

export default node;
