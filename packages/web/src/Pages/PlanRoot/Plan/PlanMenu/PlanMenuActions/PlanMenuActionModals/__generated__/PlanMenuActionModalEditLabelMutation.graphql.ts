/**
 * @generated SignedSource<<0d328156ff9e79b7a5e44e33c9cd7af1>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, FragmentRefs } from 'relay-runtime';
export type UserPlanUpdateInput = {
  planId: string;
  setLabel?: string | null | undefined;
  userId: string;
};
export type PlanMenuActionModalEditLabelMutation$variables = {
  input: UserPlanUpdateInput;
};
export type PlanMenuActionModalEditLabelMutation$data = {
  readonly userPlanUpdate: {
    readonly slug: string;
    readonly " $fragmentSpreads": FragmentRefs<"PlanWithoutParamsFragment">;
  };
};
export type PlanMenuActionModalEditLabelMutation = {
  response: PlanMenuActionModalEditLabelMutation$data;
  variables: PlanMenuActionModalEditLabelMutation$variables;
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
  "name": "slug",
  "storageKey": null
};
return {
  "fragment": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Fragment",
    "metadata": null,
    "name": "PlanMenuActionModalEditLabelMutation",
    "selections": [
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": "PlanWithHistory",
        "kind": "LinkedField",
        "name": "userPlanUpdate",
        "plural": false,
        "selections": [
          (v2/*: any*/),
          {
            "args": null,
            "kind": "FragmentSpread",
            "name": "PlanWithoutParamsFragment"
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
    "name": "PlanMenuActionModalEditLabelMutation",
    "selections": [
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": "PlanWithHistory",
        "kind": "LinkedField",
        "name": "userPlanUpdate",
        "plural": false,
        "selections": [
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
            "name": "label",
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
        ],
        "storageKey": null
      }
    ]
  },
  "params": {
    "cacheID": "c7309a5b7c885e882a017840fc5ecf3e",
    "id": null,
    "metadata": {},
    "name": "PlanMenuActionModalEditLabelMutation",
    "operationKind": "mutation",
    "text": "mutation PlanMenuActionModalEditLabelMutation(\n  $input: UserPlanUpdateInput!\n) {\n  userPlanUpdate(input: $input) {\n    slug\n    ...PlanWithoutParamsFragment\n    id\n  }\n}\n\nfragment PlanWithoutParamsFragment on PlanWithHistory {\n  id\n  isMain\n  label\n  slug\n  addedToServerAt\n  sortTime\n  lastSyncAt\n}\n"
  }
};
})();

(node as any).hash = "9b3dc4361b962566332fc76eefd982b7";

export default node;
