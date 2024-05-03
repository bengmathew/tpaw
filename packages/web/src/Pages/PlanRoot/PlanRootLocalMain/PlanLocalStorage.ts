import {
  Guards,
  PlanParams,
  PlanParamsChangeAction,
  SomePlanParams,
  SomePlanParamsVersion,
  block,
  currentPlanParamsVersion,
  fGet,
  planParamsBackwardsCompatibleGuard,
  planParamsChangeActionGuard,
  planParamsMigrate,
} from '@tpaw/common'
import {
  JSONGuard,
  chain,
  constant,
  gte,
  integer,
  json,
  number,
  object,
  string,
  nullable,
} from 'json-guard'
import _ from 'lodash'
import * as uuid from 'uuid'

type _PlanParamsHistoryItemUnmigratedUnsorted = {
  readonly id: string
  readonly change: PlanParamsChangeAction
  readonly params: SomePlanParams
}

type __PlanParamsHistoryItemMigrated = Omit<
  _PlanParamsHistoryItemUnmigratedUnsorted,
  'params'
> & {
  readonly paramsUnmigrated: SomePlanParams | null
  readonly params: PlanParams
}

export type PlanLocalStorageUnmigratedUnsorted = {
  readonly v: 1
  readonly planId: string
  readonly planParamsPostBaseUnmigratedUnsorted: readonly _PlanParamsHistoryItemUnmigratedUnsorted[]
  readonly reverseHeadIndex: number
}
export type PlanLocalStorageMigrated = Omit<
  PlanLocalStorageUnmigratedUnsorted,
  'planParamsPostBaseUnmigratedUnsorted'
> & {
  readonly planParamsPostBase: readonly __PlanParamsHistoryItemMigrated[]
}

const _stateGuard: JSONGuard<
  Omit<
    PlanLocalStorageUnmigratedUnsorted,
    'planParamsPostBaseUnmigratedUnsorted'
  >
> = object({
  v: constant(1),
  planId: Guards.uuid,
  reverseHeadIndex: chain(number, integer, gte(0)),
})

const _planParamsPostBaseItemGuard: JSONGuard<_PlanParamsHistoryItemUnmigratedUnsorted> =
  object({
    id: Guards.uuid,
    params: planParamsBackwardsCompatibleGuard,
    change: planParamsChangeActionGuard,
  })

export namespace PlanLocalStorage {
  export const getDefault = (
    planParams: PlanParams,
  ): {
    state: PlanLocalStorageMigrated
    planMigratedFromVersion: SomePlanParamsVersion
  } => {
    return {
      state: {
        v: 1,
        planId: uuid.v4(),
        planParamsPostBase: [
          {
            id: uuid.v4(),
            paramsUnmigrated: null,
            params: planParams,
            change: { type: 'start', value: null },
          },
        ],
        reverseHeadIndex: 0,
      },
      planMigratedFromVersion: currentPlanParamsVersion,
    }
  }

  export const readUnMigratedUnsorted =
    (): PlanLocalStorageUnmigratedUnsorted | null => {
      const oldParams = window.localStorage.getItem('params')
      if (oldParams) {
        const planParams = chain(
          string,
          json,
          planParamsBackwardsCompatibleGuard,
        )(oldParams).force()

        const state: PlanLocalStorageUnmigratedUnsorted = {
          v: 1,
          planId: uuid.v4(),
          planParamsPostBaseUnmigratedUnsorted: [
            {
              id: uuid.v4(),
              params: planParams,
              change: { type: 'startCopiedFromBeforeHistory', value: null },
            },
          ],
          reverseHeadIndex: 0,
        }
        write(state)
        window.localStorage.removeItem('params')
        return state
      } else {
        const str = window.localStorage.getItem('plan')
        if (!str) return null
        const stateWithoutHistory = chain(
          string,
          json,
          _stateGuard,
        )(str).force()

        return {
          ...stateWithoutHistory,
          planParamsPostBaseUnmigratedUnsorted: _.entries(localStorage)
            .filter(([key]) => _planParamsPostBaseItemKey.isKey(key))
            .map(([id, str]) =>
              chain(string, json, _planParamsPostBaseItemGuard)(str).force(),
            ),
        }
      }
    }

  export const read = (): {
    state: PlanLocalStorageMigrated
    planMigratedFromVersion: SomePlanParamsVersion
  } | null => {
    const unmigratedUnsortedState = readUnMigratedUnsorted()
    if (!unmigratedUnsortedState) return null
    const { planParamsPostBaseUnmigratedUnsorted, ...restState } =
      unmigratedUnsortedState
    const planParamsPostBase =
      unmigratedUnsortedState.planParamsPostBaseUnmigratedUnsorted
        .map((x) => ({
          id: x.id,
          params: {
            unmigrated: x.params,
            migrated: planParamsMigrate(x.params),
          },
          change: x.change,
        }))
        .sort(
          (a, b) => a.params.migrated.timestamp - b.params.migrated.timestamp,
        )
    const state = {
      ...restState,
      planParamsPostBase: planParamsPostBase.map((x) => ({
        id: x.id,
        paramsUnmigrated: x.params.unmigrated,
        params: x.params.migrated,
        change: x.change,
      })),
    }
    // Intentionally last and not the one pointer to by reverseHeadIndex.
    const lastParamsUnmigrated = fGet(_.last(planParamsPostBase)).params
      .unmigrated
    const planMigratedFromVersion =
      'v' in lastParamsUnmigrated ? lastParamsUnmigrated.v : (1 as const)
    return { state, planMigratedFromVersion }
  }

  // Ok to take in migrated since we only write the new params, which
  // should be in the latest PlanParams.
  export const write = (
    state: PlanLocalStorageMigrated | PlanLocalStorageUnmigratedUnsorted,
  ) => {
    const { planParamsPostBase, restState } = block(() => {
      if ('planParamsPostBase' in state) {
        const { planParamsPostBase, ...restState } = state
        return {
          planParamsPostBase: planParamsPostBase.map(
            (x): _PlanParamsHistoryItemUnmigratedUnsorted => ({
              id: x.id,
              params: x.params,
              change: x.change,
            }),
          ),
          restState,
        }
      } else {
        const {
          planParamsPostBaseUnmigratedUnsorted: planParamsPostBase,
          ...restState
        } = state
        return { planParamsPostBase, restState }
      }
    })

    const stateToStore: Omit<
      PlanLocalStorageMigrated | PlanLocalStorageUnmigratedUnsorted,
      'planParamsPostBase'
    > = {
      ...restState,
      v: 1,
    }
    window.localStorage.setItem(
      'plan',
      JSON.stringify(_stateGuard(stateToStore).force()),
    )

    const keysToKeep = new Set(
      planParamsPostBase.map((x) => _planParamsPostBaseItemKey.get(x.id)),
    )
    _.keys(localStorage)
      .filter((key) => _planParamsPostBaseItemKey.isKey(key))
      .filter((x) => !keysToKeep.has(x))
      .forEach((key) => localStorage.removeItem(key))

    planParamsPostBase.map((x) => {
      const key = _planParamsPostBaseItemKey.get(x.id)
      if (window.localStorage.getItem(key)) return
      window.localStorage.setItem(key, JSON.stringify(x))
    })

    // Handle hard rebasing changing the first change type to
    // "startCutByClient".
    if ('planParamsPostBase' in state) {
      const currFirst = fGet(_.first(state.planParamsPostBase))
      const firstKey = _planParamsPostBaseItemKey.get(currFirst.id)
      const storedFirst = chain(
        string,
        json,
        _planParamsPostBaseItemGuard,
      )(window.localStorage.getItem(firstKey)).force()
      if (!_.isEqual(storedFirst.change, currFirst.change)) {
        const newStored: typeof storedFirst = {
          ...storedFirst,
          change: currFirst.change,
        }
        window.localStorage.setItem(firstKey, JSON.stringify(newStored))
      }
    }
  }

  export const clear = () => {
    window.localStorage.removeItem('plan')
    _.keys(localStorage)
      .filter((key) => _planParamsPostBaseItemKey.isKey(key))
      .forEach((key) => window.localStorage.removeItem(key))
  }
}

const _planParamsPostBaseItemKey = {
  get: (id: string) => `planParamsPostBaseItem-${id}`,
  isKey: (key: string) => key.startsWith('planParamsPostBaseItem-'),
}
