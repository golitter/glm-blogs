import { BackToTop } from "@/components/BackToTop";
import { CategoryTree } from "@/components/blog/CategoryTree";
import { RecentUpdates } from "@/components/blog/RecentUpdates";
import { SearchBar } from "@/components/blog/SearchBar";
import { SiteFooter } from "@/components/layout/SiteFooter";
import { SiteHeader } from "@/components/layout/SiteHeader";
import { AboutCard } from "@/components/sidebar/AboutCard";
import { SearchTips } from "@/components/sidebar/SearchTips";

function App() {
  return (
    <div className="min-h-screen bg-canvas text-ink">
      <a
        href="#main"
        className="sr-only rounded-md focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-30 focus:bg-surface focus:px-3 focus:py-2 focus:text-sm focus:font-semibold focus:text-ink focus:shadow-lift"
      >
        跳到主要内容
      </a>

      <SiteHeader />

      <div className="mx-auto w-full max-w-[1240px] px-6 py-10 max-[1120px]:py-7">
        <div className="grid grid-cols-[224px_minmax(0,1fr)_248px] items-start gap-6 max-[1120px]:grid-cols-1">
          <aside className="sticky top-20 max-[1120px]:order-2 max-[1120px]:static max-[1120px]:top-auto">
            <AboutCard />
          </aside>

          <main id="main" className="w-full max-[1120px]:order-1">
            <h1 className="sr-only">Golemon Blogs — 技术博客导航</h1>

            <section className="mb-6">
              <SearchBar />
            </section>

            <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,320px)] items-start gap-6 max-[1120px]:grid-cols-1">
              <CategoryTree />
              <RecentUpdates />
            </div>
          </main>

          <aside className="sticky top-20 max-[1120px]:order-3 max-[1120px]:static max-[1120px]:top-auto">
            <SearchTips />
          </aside>
        </div>

        <SiteFooter />
      </div>

      <BackToTop />
    </div>
  );
}

export default App;
