import {
  Learn,
  learnGetStaticProps,
} from '../../src/Pages/Learn/Learn'
import {Contentful} from '../../src/Utils/Contentful'

export default Learn

export const getStaticProps = learnGetStaticProps

export async function getStaticPaths() {
  const slugs = await Contentful.getKnowledgeBaseArticleSlugs()

  return {
    paths: slugs.map(slug => ({
      params: {slug: slug === 'knowledge-base-start' ? [] : [slug]},
    })),
    fallback: false,
  }
}