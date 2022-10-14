import {default as React, ReactElement, ReactNode} from 'react'
import {PlanInputType} from '../Helpers/PlanInputType'
import {PlanInputSizing} from '../PlanInput'
import {PlanInputBodyMobile} from './PlanInputBodyMobile'
import {PlanInputLaptopAndDesktop} from './PlanInputLaptopAndDesktop'

export type PlanInputBodyPassThruProps = Omit<
  React.ComponentProps<typeof PlanInputBody>,
  'children'
>

export const PlanInputBody = React.memo(
  ({
    layout,
    sizing,
    children: childrenIn,
    type,
    customGuideIntro,
  }: {
    layout: 'mobile' | 'laptop' | 'desktop'
    sizing: PlanInputSizing['fixed']
    type: PlanInputType
    customGuideIntro?: ReactNode
    children:
      | ReactElement
      | [
          ReactElement,
          {
            error?: ReactElement
            input?: (
              transitionOut: (onDone: () => void) => void
            ) => ReactElement
          }
        ]
  }) => {
    const [content, children] =
      childrenIn instanceof Array ? childrenIn : ([childrenIn, null] as const)

    const child = {...children, content}

    return layout === 'mobile' ? (
      <PlanInputBodyMobile
        sizing={sizing}
        type={type}
        customGuideIntro={customGuideIntro}
      >
        {child}
      </PlanInputBodyMobile>
    ) : (
      <PlanInputLaptopAndDesktop
        sizing={sizing}
        layout={layout}
        type={type}
        customGuideIntro={customGuideIntro}
      >
        {child}
      </PlanInputLaptopAndDesktop>
    )
  }
)
