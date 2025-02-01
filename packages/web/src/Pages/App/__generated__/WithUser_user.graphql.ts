/**
 * @generated SignedSource<<b9b775eebea1f5fc9883f2b67a5033dd>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ReaderFragment } from 'relay-runtime';
import { FragmentRefs } from "relay-runtime";
export type WithUser_user$data = {
  readonly id: string;
  readonly nonPlanParams: string;
  readonly nonPlanParamsLastUpdatedAt: number;
  readonly plans: ReadonlyArray<{
    readonly addedToServerAt: number;
    readonly id: string;
    readonly isDated: boolean;
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
        },
        {
          "alias": null,
          "args": null,
          "kind": "ScalarField",
          "name": "isDated",
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

(node as any).hash = "656c9e90d2344967067e6cdb514ed774";

export default node;
