import _ from 'lodash'
import { assert, fGet } from '../../../Utils/Utils'
import {
  PlanSectionName,
  planSectionNames,
} from './PlanInput/Helpers/PlanSectionName'

export type PlanTransitionState = {
  section: PlanSectionName
  dialogMode: boolean
}

export type PlanTransition = {
  prev: PlanTransitionState
  target: PlanTransitionState
  progress: number
}

type _SimpleState<Label extends string> = {
  label: Label
  sections: readonly {
    section: PlanSectionName | 'rest'
    dialogMode: boolean | 'any'
  }[]
}

type _SimpleStateNormalized<Label extends string> = {
  label: Label
  sections: readonly {
    section: PlanSectionName
    dialogMode: boolean
  }[]
}

const _allStates: readonly PlanTransitionState[] = _.flatten(
  planSectionNames.map((section) => [
    { section, dialogMode: false, chartHover: false },
    { section, dialogMode: false, chartHover: true },
    { section, dialogMode: true, chartHover: false },
    { section, dialogMode: true, chartHover: true },
  ]),
)
export const _toSimpleStatesNormalized = (
  simpleStates: _SimpleState<string>[],
): _SimpleStateNormalized<string>[] => {
  const statesSoFar: _SimpleStateNormalized<string>[] = []
  simpleStates.forEach((simpleState) => {
    statesSoFar.push({
      label: simpleState.label,
      sections: _.flatten(
        simpleState.sections.map(({ section, dialogMode }) => {
          const dialogModes =
            dialogMode === 'any' ? [true, false] : [dialogMode]
          const result: _SimpleStateNormalized<never>['sections'][0][] = []
          dialogModes.forEach((dialogMode) => {
            if (section !== 'rest') {
              result.push({ section, dialogMode })
            } else {
              result.push(
                ..._.differenceWith(
                  _allStates,
                  _.flatten(statesSoFar.map((x) => x.sections)),
                  _.isEqual,
                ).filter((x) => x.dialogMode === dialogMode),
              )
            }
          })
          return result
        }),
      ),
    })
  })
  const allStates = _.flatten(statesSoFar.map((x) => x.sections))
  assert(_.uniqWith(allStates, _.isEqual).length === allStates.length)
  return statesSoFar
}

export const _planStateToSimpleState = (
  { section, dialogMode }: PlanTransitionState,
  simpleStates: _SimpleStateNormalized<string>[],
) =>
  fGet(
    simpleStates.find(
      (x) => _.find(x.sections, { section: section, dialogMode }) !== undefined,
    ),
  ).label

export const simplifyPlanTransitionState2 = <
  S1 extends string,
  S2 extends string,
>(
  state1: _SimpleState<S1>,
  state2: _SimpleState<S2>,
) => {
  const normStates = _toSimpleStatesNormalized([state1, state2])
  return (x: PlanTransitionState) =>
    _planStateToSimpleState(x, normStates) as S1 | S2
}

export const simplifyPlanTransitionState3 = <
  S1 extends string,
  S2 extends string,
  S3 extends string,
>(
  state1: _SimpleState<S1>,
  state2: _SimpleState<S2>,
  state3: _SimpleState<S3>,
) => {
  const normStates = _toSimpleStatesNormalized([state1, state2, state3])
  return (x: PlanTransitionState) =>
    _planStateToSimpleState(x, normStates) as S1 | S2 | S3
}

export const simplifyPlanTransitionState4 = <
  S1 extends string,
  S2 extends string,
  S3 extends string,
  S4 extends string,
>(
  state1: _SimpleState<S1>,
  state2: _SimpleState<S2>,
  state3: _SimpleState<S3>,
  state4: _SimpleState<S4>,
) => {
  const normStates = _toSimpleStatesNormalized([state1, state2, state3, state4])
  return (x: PlanTransitionState) =>
    _planStateToSimpleState(x, normStates) as S1 | S2 | S3 | S4
}

export const simplifyPlanTransitionState5 = <
  S1 extends string,
  S2 extends string,
  S3 extends string,
  S4 extends string,
  S5 extends string,
>(
  state1: _SimpleState<S1>,
  state2: _SimpleState<S2>,
  state3: _SimpleState<S3>,
  state4: _SimpleState<S4>,
  state5: _SimpleState<S5>,
) => {
  const normStates = _toSimpleStatesNormalized([
    state1,
    state2,
    state3,
    state4,
    state5,
  ])
  return (x: PlanTransitionState) =>
    _planStateToSimpleState(x, normStates) as S1 | S2 | S3 | S4 | S5
}
