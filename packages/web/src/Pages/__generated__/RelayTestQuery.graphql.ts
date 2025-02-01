/**
 * @generated SignedSource<<e74ce8925741d683704d5c8dcc7ee1e7>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest } from 'relay-runtime';
export type RelayTestQuery$variables = Record<PropertyKey, never>;
export type RelayTestQuery$data = {
  readonly ping: string;
};
export type RelayTestQuery = {
  response: RelayTestQuery$data;
  variables: RelayTestQuery$variables;
};

const node: ConcreteRequest = (function(){
var v0 = [
  {
    "alias": null,
    "args": null,
    "kind": "ScalarField",
    "name": "ping",
    "storageKey": null
  }
];
return {
  "fragment": {
    "argumentDefinitions": [],
    "kind": "Fragment",
    "metadata": null,
    "name": "RelayTestQuery",
    "selections": (v0/*: any*/),
    "type": "Query",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": [],
    "kind": "Operation",
    "name": "RelayTestQuery",
    "selections": (v0/*: any*/)
  },
  "params": {
    "cacheID": "601487fd90209ec485811c9dce07429a",
    "id": null,
    "metadata": {},
    "name": "RelayTestQuery",
    "operationKind": "query",
    "text": "query RelayTestQuery {\n  ping\n}\n"
  }
};
})();

(node as any).hash = "94998321309c9b1729c81a04ceaaeff2";

export default node;
