import {newPadding, rectExt} from '../../../Utils/Geometry'
import {linearFnFomPoints} from '../../../Utils/LinearFn'
import {PlanSizing} from './PlanSizing'

const pad = 40
const cardPadding = newPadding(20)

export function planSizingDesktop(windowSize: {
  width: number
  height: number
}): PlanSizing {

  const contentWidth = 600

  // ---- WELCOME ----
  const welcome = ((): PlanSizing['welcome'] => {
    const width = 500
    const inOriginX = (windowSize.width - width) / 2
    return {
      dynamic: {
        in: {
          origin: {x: inOriginX, y: 0},
          opacity: 1,
        },
        out: {
          origin: {x: inOriginX - 25, y: 0},
          opacity: 0,
        },
      },
      fixed: {
        size: {width, height: windowSize.height},
      },
    }
  })()

  // ---- CHART ----

  const chart = ((): PlanSizing['chart'] => {
    type Dynamic = PlanSizing['chart']['dynamic']['hidden']
    const summaryState: Dynamic = {
      region: rectExt({
        x: 0,
        y: 0,
        width: windowSize.width,
        height:
          windowSize.width < 800
            ? linearFnFomPoints(400, 300, 800, 550)(windowSize.width)
            : 550,
      }),
      padding: newPadding({vert: pad * 1.5, horz: pad}),
      legacyWidth: 120,
      intraGap: pad,
      borderRadius: 0,
      opacity: 1,
      tasksOpacity: 1,
    }

    const inputState = {
      ...summaryState,
      region: rectExt({
        x: 0,
        y: 0,
        width: summaryState.region.width,
        height: summaryState.region.height - 50,
      }),
      padding: newPadding({
        horz: summaryState.padding.left,
        top: summaryState.padding.top,
        bottom: summaryState.padding.bottom * 0.75,
      }),
      tasksOpacity: 0,
    }
    const resultsState: Dynamic = {
      ...inputState,
    }
    const hiddenState = {
      ...resultsState,
      region: rectExt.translate(resultsState.region, {x: 0, y: -30}),
      opacity: 0,
    }
    return {
      dynamic: {
        summary: summaryState,
        input: inputState,
        results: resultsState,
        hidden: hiddenState,
      },
      fixed: {
        cardPadding:{left: 15, right: 15, top: 10, bottom: 10},
      },
    }
  })()

  // ---- INPUT ----

  const input = ((): PlanSizing['input'] => {

    return {
      dynamic: {
        dialogModeIn: {
          origin: {x: 0, y: 0},
          opacity: 1,
        },
        dialogModeOutRight: {
          origin: {x: 25, y: 0},
          opacity: 0,
        },
        dialogModeOutLeft: {
          origin: {x: -25, y: 0},
          opacity: 0,
        },
        notDialogModeIn: {
          origin: {x: 0, y: chart.dynamic.input.region.bottom},
          opacity: 1,
        },
        notDialogModeOut: {
          origin: {x: 0, y: chart.dynamic.summary.region.bottom},
          opacity: 0,
        },
      },
      fixed: {
        dialogMode: {
          size: {...windowSize},
          padding: {horz: (windowSize.width - contentWidth) / 2, top: pad * 2},
        },
        notDialogMode: {
          size: {
            width: windowSize.width,
            height: windowSize.height - chart.dynamic.input.region.bottom,
          },
          padding: {
            left: pad,
            right: windowSize.width - contentWidth - pad,
            top: pad,
          },
        },
        cardPadding,
      },
    }
  })()

  // ---- SUMMARY ----
  const summary = ((): PlanSizing['summary'] => {
    return {
      dynamic: {
        in: {
          origin: {x: 0, y: chart.dynamic.summary.region.bottom},
          opacity: 1,
        },
        out: {
          origin: {x: 0, y: chart.dynamic.input.region.bottom},
          opacity: 0,
        },
      },
      fixed: {
        size: {
          width: windowSize.width,
          height: windowSize.height - chart.dynamic.summary.region.height,
        },
        padding: newPadding({
          left: pad,
          right: windowSize.width - contentWidth - pad,
          top: pad / 2,
          bottom: 0,
        }),
        cardPadding,
      },
    }
  })()

  // ---- RESULTS ----
  const results = ((): PlanSizing['results'] => {
    return {
      dynamic: {
        in: {
          origin: {x: 0, y: chart.dynamic.results.region.bottom},
          opacity: 1,
        },
        outDialogMode: {
          origin: {x: 0, y: chart.dynamic.results.region.bottom + 30},
          opacity: 0,
        },
        outNotDialogMode: {
          origin: {x: 0, y: chart.dynamic.summary.region.bottom},
          opacity: 0,
        },
      },
      fixed: {
        size: {
          width: windowSize.width,
          height: windowSize.height - chart.dynamic.results.region.bottom,
        },
        padding: input.fixed.notDialogMode.padding,
      },
    }
  })()

  return {welcome, chart, input, results, summary}
}
