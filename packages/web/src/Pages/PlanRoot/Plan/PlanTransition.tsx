import _ from 'lodash'
import {assert, fGet} from '../../../Utils/Utils'
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
    name: PlanSectionName | 'rest'
    dialogMode: boolean | 'any'
  }[]
}

type _SimpleStateNormalized<Label extends string> = {
  label: Label
  sections: readonly {name: PlanSectionName; dialogMode: boolean}[]
}

export const _toSimpleStatesNormalized = (
  simpleStates: _SimpleState<string>[]
): _SimpleStateNormalized<string>[] => {
  const statesSoFar = [] as {name: PlanSectionName; dialogMode: boolean}[]
  const getRestSections = (dialogMode: 'any' | boolean) => {
    const [statesSoFarDialogMode, statesSoFarNotDialogMode] = _.partition(
      statesSoFar,
      x => x.dialogMode
    ).map(x => x.map(x => x.name))
    const restDialogMode = _.difference(
      planSectionNames,
      statesSoFarDialogMode
    ).map(name => ({name, dialogMode: true}))
    const restNotDialogMode = _.difference(
      planSectionNames,
      statesSoFarNotDialogMode
    ).map(name => ({name, dialogMode: false}))
    return dialogMode === 'any'
      ? [...restDialogMode, ...restNotDialogMode]
      : dialogMode
      ? restDialogMode
      : restNotDialogMode
  }

  const result = simpleStates.map(simpleState => {
    const sections = _.flatten(
      simpleState.sections.map(x =>
        x.name === 'rest'
          ? getRestSections(x.dialogMode)
          : x.dialogMode === 'any'
          ? [
              {name: x.name, dialogMode: true},
              {name: x.name, dialogMode: false},
            ]
          : [{name: x.name, dialogMode: x.dialogMode as boolean}]
      )
    )
    assert(_.intersectionWith(statesSoFar, sections, _.isEqual))
    statesSoFar.push(...sections)
    return {label: simpleState.label, sections}
  })
  assert(statesSoFar.length === planSectionNames.length * 2)
  return result
}

export const _planStateToSimpleState = (
  {section, dialogMode}: PlanTransitionState,
  simpleStates: _SimpleStateNormalized<string>[]
) =>
  fGet(
    simpleStates.find(
      x => _.find(x.sections, {name: section, dialogMode}) !== undefined
    )
  ).label



export const simplifyPlanTransitionState2 = <
  S1 extends string,
  S2 extends string
>(
  state1: _SimpleState<S1>,
  state2: _SimpleState<S2>
) => {
  const normStates = _toSimpleStatesNormalized([state1, state2])
  return (x: PlanTransitionState) =>
    _planStateToSimpleState(x, normStates) as S1 | S2
}

export const simplifyPlanTransitionState3 = <
  S1 extends string,
  S2 extends string,
  S3 extends string
>(
  state1: _SimpleState<S1>,
  state2: _SimpleState<S2>,
  state3: _SimpleState<S3>
) => {
  const normStates = _toSimpleStatesNormalized([state1, state2, state3])
  return (x: PlanTransitionState) =>
    _planStateToSimpleState(x, normStates) as S1 | S2 | S3
}

export const simplifyPlanTransitionState4 = <
  S1 extends string,
  S2 extends string,
  S3 extends string,
  S4 extends string
>(
  state1: _SimpleState<S1>,
  state2: _SimpleState<S2>,
  state3: _SimpleState<S3>,
  state4: _SimpleState<S4>
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
  S5 extends string
>(
  state1: _SimpleState<S1>,
  state2: _SimpleState<S2>,
  state3: _SimpleState<S3>,
  state4: _SimpleState<S4>,
  state5: _SimpleState<S5>
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
