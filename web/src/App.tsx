import { useEffect, useState } from 'react';
import CaseList from './pages/CaseList';
import Inspect from './pages/Inspect';
import About from './pages/About';

function useHashPath() {
  const read = () => window.location.hash.replace(/^#/, '') || '/';
  const [path, setPath] = useState(read);
  useEffect(() => {
    const on = () => setPath(read());
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return path;
}

export default function App() {
  const path = useHashPath();
  const caseId = path.match(/^\/case\/([^/]+)/)?.[1];
  const about = path.startsWith('/about');

  let page;
  if (caseId) page = <Inspect key={caseId} id={decodeURIComponent(caseId)} />;
  else if (about) page = <About />;
  else page = <CaseList />;

  return (
    <div className="shell">
      <header className="topbar">
        <span className="brand">스캔 검사</span>
        <nav>
          <a href="#/" className={!about ? 'on' : ''}>검사 작업</a>
          <a href="#/about" className={about ? 'on' : ''}>모델 · 검증 결과</a>
        </nav>
        <span className="spacer" />
        <span className="note">데모 · 합성 데이터로 학습한 모델 · 이미지는 브라우저 밖으로 전송되지 않습니다</span>
      </header>
      <main>{page}</main>
    </div>
  );
}
