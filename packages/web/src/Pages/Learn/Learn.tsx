import { faChevronUp, faStream } from '@fortawesome/pro-regular-svg-icons'
import { faChevronRight, faHomeAlt } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Transition } from '@headlessui/react'
import _ from 'lodash'
import { GetStaticProps, InferGetStaticPropsType } from 'next'
import Link from 'next/link'
import { useRouter } from 'next/router'
import React, { useEffect, useState } from 'react'
import ReactDOM from 'react-dom'
import { Contentful } from '../../Utils/Contentful'
import { assert } from '../../Utils/Utils'
import { AppPage } from '../App/AppPage'
import { mainPlanColors } from '../PlanRoot/Plan/UsePlanColors'

export const learnGetStaticProps: GetStaticProps<{
  outline: Contentful.FetchedKnowledgeBaseOutline
  content: Contentful.FetchedKnowledgeBaseArticle
}> = async ({ params }) => {
  const slugArr = params?.slug ?? []
  assert(slugArr instanceof Array)
  if (slugArr.length > 1) return { notFound: true }
  const slug = slugArr.length > 0 ? slugArr[0] : 'knowledge-base-start' // ignore everything but last

  return {
    props: {
      outline: await Contentful.fetchKnowledgeBaseOutline(
        '4QWP7hVZ4lZ3ZD8Bz1nWH3',
      ),
      content: await Contentful.fetchKnowledgeBaseArticle(slug),
    },
  }
}

export const Learn = React.memo(
  (props: InferGetStaticPropsType<typeof learnGetStaticProps>) => {
    const { content } = props
    return (
      <AppPage
        className="min-h-screen"
        title={`${content.fields.publicTitle} - Knowledge Base - TPAW Planner`}
      >
        <>
          <_Desktop {...props} />
          <_Mobile {...props} />
        </>
      </AppPage>
    )
  },
)

const _Desktop = React.memo(
  ({
    outline,
    content,
  }: InferGetStaticPropsType<typeof learnGetStaticProps>) => {
    return (
      <div
        className={` min-h-screen hidden learn:grid`}
        style={{ grid: '1fr auto/ 1fr 2fr' }}
        tabIndex={0}
      >
        <div className="bg-gray-100 h-screen py-4 px-8 sticky top-0 pt-header overflow-scroll">
          <_Outline
            className="mt-6"
            outline={outline}
            slug={content.fields.slug}
            onClick={() => {}}
          />
        </div>
        <div className="pl-8 relative pt-header max-w-[650px] ">
          <_Content
            className="relative z-0 pr-8 mb-20  mt-4"
            {...{ content, outline }}
          />
        </div>
      </div>
    )
  },
)

const _Mobile = React.memo(
  ({
    outline,
    content,
  }: InferGetStaticPropsType<typeof learnGetStaticProps>) => {
    const [showContents, setShowContents] = useState(false)
    return (
      <div className={` learn:hidden  min-h-screen pt-header`}>
        <_Content
          className="relative z-0 px-4 sm:px-8 mb-20 mt-6"
          {...{ content, outline }}
        />
        <button
          className="fixed bottom-0 right-0 mb-4 mr-4 btn-dark btn-lg flex items-center gap-x-2"
          style={{ boxShadow: '0px 0px 15px 5px rgba(0,0,0,0.28)' }}
          onClick={() => setShowContents(true)}
        >
          <FontAwesomeIcon className="text-lg" icon={faStream} />
          Contents
          <FontAwesomeIcon className="text-lg" icon={faChevronUp} />
        </button>
        <_MobileOutline
          {...{ outline, content, showContents, setShowContents }}
        />
      </div>
    )
  },
)

const _MobileOutline = React.memo(
  ({
    outline,
    showContents,
    setShowContents,
    content,
  }: {
    outline: Contentful.FetchedKnowledgeBaseOutline
    content: Contentful.FetchedKnowledgeBaseArticle
    showContents: boolean
    setShowContents: (x: boolean) => void
  }) => {
    return ReactDOM.createPortal(
      <Transition
        className="page"
        show={showContents}
        onClick={() => setShowContents(false)}
      >
        <Transition.Child
          className="fixed inset-0 bg-black/50 "
          enter="transition-opacity duration-300"
          enterFrom="opacity-0"
          leave="transition-opacity duration-300"
          leaveTo="opacity-0"
        />

        <Transition.Child
          className="fixed bottom-0 right-0 w-[600px] max-w-[100vw]  rounded-tl-xl  rounded-tr-xl bg-pageBG  max-h-[70vh] overflow-scroll"
          enter="transition-transfrom duration-300"
          enterFrom=" translate-y-[200px] opacity-0"
          leave="transition-transfrom duration-300"
          leaveTo=" translate-y-[200px] opacity-0"
          style={{ boxShadow: '0px 0px 10px 5px rgba(0,0,0,0.28)' }}
        >
          <_Outline
            className=" m-3"
            outline={outline}
            slug={content.fields.slug}
            onClick={() => setShowContents(false)}
          />
        </Transition.Child>
      </Transition>,
      window.document.body,
    )
  },
)

const _Content = React.memo(
  ({
    className = '',
    content,
    outline,
  }: {
    className?: string
    content: Contentful.FetchedKnowledgeBaseArticle
    outline: Contentful.FetchedKnowledgeBaseOutline
  }) => {
    const flatOutline = _flatten(outline)
    const lastIndex = _.findLastIndex(
      flatOutline,
      (x) => x.slug === content.fields.slug,
    )
    const firstIndex = _.findIndex(
      flatOutline,
      (x) => x.slug === content.fields.slug,
    )
    const next =
      lastIndex === flatOutline.length - 1 ? null : flatOutline[lastIndex + 1]
    const prev = firstIndex === 0 ? null : flatOutline[firstIndex - 1]
    const parentChain = _parentChain(flatOutline, lastIndex)

    const router = useRouter()
    useEffect(() => {
      const handler = (e: KeyboardEvent) => {
        if (e.key === 'ArrowRight' && next) {
          void router.push(_href(next.slug))
        }
        if (e.key === 'ArrowLeft' && prev) {
          void router.push(_href(prev.slug))
        }
      }
      window.addEventListener('keydown', handler)
      return () => window.removeEventListener('keydown', handler)
    }, [next, prev, router])

    return (
      <div className={`${className} `}>
        {parentChain.length > 0 && (
          <div className="mt-2  gap-x-2">
            <Link className="font-bold text-lg mr-4" key={'home'} href="/learn">
              <FontAwesomeIcon icon={faHomeAlt} />
            </Link>
            {parentChain.map((x, i) => (
              <React.Fragment key={i}>
                <FontAwesomeIcon
                  className="lighten-2 mr-4 text-sm"
                  icon={faChevronRight}
                />
                <Link className="font-bold text-lg mr-4" href={x.slug}>
                  {x.title}
                </Link>
              </React.Fragment>
            ))}
          </div>
        )}
        <h1 className="text-4xl font-bold mb-6 mt-2">
          {content.fields.publicTitle}
        </h1>
        <Contentful.RichText
          body={content.fields.body}
          h2="font-bold text-xl mt-6"
          p="my-4 p-base"
          p6=" p-base"
          ol="list-decimal ml-5"
        />
        {next && (
          <div
            className="grid mt-10"
            style={{ grid: '"prev middle next" auto / auto 1fr auto' }}
          >
            <Link
              className="flex flex-col items-end "
              style={{ gridArea: 'next' }}
              href={next.slug}
            >
              <h2 className="">
                Next <FontAwesomeIcon className="" icon={faChevronRight} />{' '}
              </h2>
              <h2 className="font-semibold">{next.title}</h2>
            </Link>
          </div>
        )}
      </div>
    )
  },
)

const _Outline = React.memo(
  ({
    className = '',
    outline,
    slug,
    onClick,
  }: {
    className?: string
    outline: Contentful.FetchedKnowledgeBaseOutline
    slug: string
    onClick: () => void
  }) => {
    return (
      <div className={`${className}`} onClick={onClick}>
        <div className="">
          {outline.fields.items.map((item, i) => {
            if (_isArticle(item)) {
              return (
                <_Link
                  key={i}
                  slug={item.fields.slug}
                  title={item.fields.publicTitle}
                  isCurrent={item.fields.slug === slug}
                  isChildCurrent={false}
                />
              )
            } else {
              const childSlugs = _flatten(item).map((x) => x.slug)
              return (
                <div className="" key={i}>
                  <_Link
                    slug={childSlugs[0]}
                    title={item.fields.title}
                    isCurrent={false}
                    isChildCurrent={childSlugs.includes(slug)}
                  />
                  <_Outline
                    className="ml-8"
                    outline={item}
                    slug={slug}
                    onClick={() => {}}
                  />
                </div>
              )
            }
          })}
        </div>
      </div>
    )
  },
)

const _Link = React.memo(
  ({
    slug,
    title,
    isCurrent,
    isChildCurrent,
  }: {
    slug: string
    title: string
    isCurrent: boolean
    isChildCurrent: boolean
  }) => (
    <Link className={`flex items-center gap-x-1 py-2 `} href={_href(slug)}>
      <FontAwesomeIcon
        className={`text-sm  ${isCurrent ? '' : 'opacity-0'}`}
        icon={faChevronRight}
      />
      <span
        className={`font-semibold`}
        style={{
          color: isCurrent ? mainPlanColors.shades.main[8].hex : undefined,
        }}
      >
        {title}
      </span>
    </Link>
  ),
)

type FlatNode = { title: string; slug: string; indent: number }
const _flatten = (
  outline: Contentful.FetchedKnowledgeBaseOutline,
  indent = 0,
): FlatNode[] => {
  return _.flatten(
    outline.fields.items.map((item) => {
      if (_isArticle(item))
        return [
          { title: item.fields.publicTitle, slug: item.fields.slug, indent },
        ]
      const children = _flatten(item, indent + 1)
      return [
        { title: item.fields.title, slug: children[0].slug, indent },
        ...children,
      ]
    }),
  )
}

const _parentChain = (outline: FlatNode[], currIndex: number): FlatNode[] => {
  const curr = outline[currIndex]
  if (curr.indent === 0) return []
  const parentIndex = _.findLastIndex(
    outline,
    (x) => x.indent === curr.indent - 1,
    currIndex,
  )
  return [..._parentChain(outline, parentIndex), outline[parentIndex]]
}

const _isArticle = (
  item:
    | Contentful.FetchedKnowledgeBaseOutline
    | Contentful.FetchedKnowledgeBaseArticle,
): item is Contentful.FetchedKnowledgeBaseArticle =>
  'publicTitle' in item.fields

const _href = (slug: string) =>
  `/learn/${slug === 'knowledge-base-start' ? '' : slug}`
