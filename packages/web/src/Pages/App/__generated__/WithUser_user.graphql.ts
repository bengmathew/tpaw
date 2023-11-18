/**
 * @generated SignedSource<<2fb7c70276838fd0b2f7208d7aba5658>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { Fragment, ReaderFragment } from 'relay-runtime';
import { FragmentRefs } from "relay-runtime";
export type WithUser_user$data = {
  readonly id: string;
  readonly nonPlanParams: string;
  readonly nonPlanParamsLastUpdatedAt: number;
  readonly plans: ReadonlyArray<{
    readonly addedToServerAt: number;
    readonly id: string;
    readonly isMain: boolean;
    readonly label: string | null | undefined;
    readonly lastSyncAt: number;
    readonly slug: string;
    readonly sortTime: number;
  }>;
  readonly " $fragmentType": "WithUser_user";
};
export type WithUser_user$key = {
  readonly " $data"?: WithUser_user$data;
  readonly " $fragmentSpreads": FragmentRefs<"WithUser_user">;
};

const node: ReaderFragment = (function(){
var v0 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "id",
  "storageKey": null
};
return {
  "argumentDefinitions": [],
  "kind": "Fragment",
  "metadata": null,
  "name": "WithUser_user",
  "selections": [
    (v0/*: any*/),
    {
      "alias": null,
      "args": null,
      "concreteType": "PlanWithHistory",
      "kind": "LinkedField",
      "name": "plans",
      "plural": true,
      "selections": [
        (v0/*: any*/),
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
        },
        {
          "alias": null,
          "args": null,
          "kind": "ScalarField",
          "name": "lastSyncAt",
          "storageKey": null
        },
        {
          "alias": null,
          "args": null,
          "kind": "ScalarField",
          "name": "isMain",
          "storageKey": null
        }
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
    }
  ],
  "type": "User",
  "abstractKey": null
};
})();

(node as any).hash = "6fc9266f1b39eba1e067461a51399607";

export default node;
