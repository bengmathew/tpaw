import { Document } from '@contentful/rich-text-types'
import React, { ReactNode, useState } from 'react'
import { Padding, paddingCSSStyle } from '../../../../../../Utils/Geometry'
import { CenteredModal } from '../../../../../Common/Modal/CenteredModal'
import { PlanInputBodyGuideContent } from './PlanInputBodyGuideContent'

export const PlanInputBodyGuide = React.memo(
  ({
    className = '',
    padding,
    guideContent,
    customIntro = `This is an advanced section. Here is a guide to help you understand how
    to use it.`,
  }: {
    className?: string
    padding: Padding
    guideContent: Document
    customIntro?: ReactNode
  }) => {
    const [showGuide, setShowGuide] = useState(false)

    return (
      <div
        className={`${className} border border-gray-400 rounded-2xl p-base`}
        style={{ ...paddingCSSStyle(padding) }}
      >
        {customIntro}{' '}
        <button className="underline" onClick={() => setShowGuide(true)}>
          Open Guide
        </button>
        <CenteredModal
          className=" dialog-outer-div"
          show={showGuide}
          onOutsideClickOrEscape={() => setShowGuide(false)}
        >
          <div className="px-2 py-4">
            <h2 className="text-2xl mb-4 w-full font-bold text-center">
              Guide
            </h2>
            <PlanInputBodyGuideContent content={guideContent} />
          </div>
        </CenteredModal>
      </div>
    )
  },
)
