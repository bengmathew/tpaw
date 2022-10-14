import {Document} from '@contentful/rich-text-types'
import {faXmark} from '@fortawesome/pro-light-svg-icons'
import {faChevronDown, faChevronRight} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React, {useMemo, useState} from 'react'
import {Contentful} from '../../../../../Utils/Contentful'
import {assert} from '../../../../../Utils/Utils'
import {usePlanInputGuideContent} from './UsePlanInputGuideContent'

export const PlanInputBodyGuideContent = React.memo(
  ({
    content,
  }: {
    content: Exclude<ReturnType<typeof usePlanInputGuideContent>, null>
  }) => {
    const {main, theory} = useMemo(
      () => _splitDocumentOnTheory(content),
      [content]
    )

    const [showTheory, setShowTheory] = useState(false)
    return (
      <div className="">
        <_Part className="" body={main} />
        {theory && (
          <div className="mt-3">
            {!showTheory && (
              <button
                className="font-italic text-sm  mb-3 "
                onClick={() => setShowTheory(!showTheory)}
              >
                Learn more
                {/* <FontAwesomeIcon
                className="ml-3 text-sm"
                icon={showTheory ? faChevronDown : faChevronRight}
              /> */}
              </button>
            )}
            {showTheory && (
              <div className="border-blue-500 bg-blue-50 bg-opacity-50 border-2 rounded-xl p-2">
                <div className="flex justify-end items-center">
                  {/* <h2 className="font-bold">Learn more</h2> */}
                  <button
                    className="text-3xl px-2"
                    onClick={() => setShowTheory(false)}
                  >
                    <FontAwesomeIcon icon={faXmark} />
                  </button>
                </div>
                <_Part className="" body={theory} />
              </div>
            )}
          </div>
        )}
      </div>
    )
  }
)

const _Part = React.memo(
  ({className = '', body}: {className?: string; body: Document}) => {
    const {intro, sections} = useMemo(
      () => _splitDocumentOnCollapse(body),
      [body]
    )
    return (
      <div className={`${className}`}>
        {intro && <_RichText className="" body={intro} />}
        {sections.map((section, index) => (
          <_Collapsable
            key={index}
            className={intro || index > 0 ? 'mt-6' : ''}
            section={section}
          />
        ))}
      </div>
    )
  }
)
const _Collapsable = React.memo(
  ({
    className = '',
    section: {body, heading},
  }: {
    className?: string
    section: {heading: string; body: Document}
  }) => {
    const [show, setShow] = useState(false)
    return (
      <div className={`${className}`}>
        <button
          className="font-bold text-lg mb-3 text-start"
          onClick={() => setShow(!show)}
        >
          {heading}
          <FontAwesomeIcon
            className="ml-3 text-sm"
            icon={show ? faChevronDown : faChevronRight}
          />
        </button>
        {show && <_RichText className="" body={body} />}
      </div>
    )
  }
)

const _RichText = React.memo(
  ({className = '', body}: {className?: string; body: Document}) => {
    return (
      <div className={`${className}`}>
        <Contentful.RichText
          body={body}
          ul="list-disc ml-5"
          ol="list-decimal ml-5"
          p="p-base mb-3"
          h2={([index]) =>
            `font-bold text-lg mb-3 ${index === 0 ? '' : 'mt-6'}`
          }
        />
      </div>
    )
  }
)

const _splitDocumentOnTheory = (document: Document) => {
  const {intro, sections} = Contentful.splitDocument(document, 'theory')
  assert(intro)
  assert(sections.length <= 1)
  return {
    main: intro,
    theory: _.first(sections)?.body ?? null,
  }
}

const _splitDocumentOnCollapse = (document: Document) =>
  Contentful.splitDocument(document, 'collapse')
