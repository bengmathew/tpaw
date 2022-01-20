import { Article, articleGetStaticProps } from '../../src/Pages/Article/Article'
import { Contentful } from '../../src/Utils/Contentful'

export default Article

export const getStaticProps = articleGetStaticProps

export async function getStaticPaths() {
  const slugs = await Contentful.getArticleSlugs()
  return {
    paths: slugs.map(slug => ({params: {slug}})),
    fallback: false,
  }
}
