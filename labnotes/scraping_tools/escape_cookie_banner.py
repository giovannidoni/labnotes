from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# --- Known CMP selectors (fast-path) ---
KNOWN_SELECTORS = [
    "#onetrust-accept-btn-handler",
    "#onetrust-reject-all-handler",  # OneTrust
    "#didomi-notice-agree-button",
    "#didomi-notice-disagree-button",  # Didomi
    "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",  # Cookiebot allow all
    "#CybotCookiebotDialogBodyButtonDecline",  # Cookiebot decline
    ".qc-cmp2-summary-buttons .qc-cmp2-summary-accept-all",  # Quantcast
    "button[mode='primary'][aria-label*='Accept' i]",
    "[data-testid='uc-accept-all-button']",
    "[data-testid='uc-reject-all-button']",
]

# Keywords to match button text in multiple languages
BTN_KEYWORDS = [
    "accept",
    "agree",
    "allow",
    "ok",
    "got it",
    "continue",
    "reject",
    "decline",
    "accetta",
    "rifiuta",  # it
    "aceptar",
    "rechazar",  # es
    "akzeptieren",
    "ablehnen",  # de
    "accepter",
    "refuser",  # fr
]


def _xpath_for_keywords():
    lowered = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    uppered = "abcdefghijklmnopqrstuvwxyz"
    parts = [f"contains(translate(normalize-space(.),'{lowered}','{uppered}'),'{kw}')" for kw in BTN_KEYWORDS]
    return " or ".join(parts)


def _click_first(driver, selectors):
    for sel in selectors:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            driver.execute_script("arguments[0].click();", el)
            return True
        except Exception:
            continue
    return False


def _click_by_text(driver, timeout=2):
    try:
        btn = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    f"//button[{_xpath_for_keywords()}] | //a[@role='button' or contains(@class,'btn')][{_xpath_for_keywords()}]",
                )
            )
        )
        driver.execute_script("arguments[0].click();", btn)
        return True
    except TimeoutException:
        return False


def _click_in_shadow_dom(driver):
    # Traverse open shadow roots, find candidate buttons by text, click the first match
    script = f"""
    const match = (el) => {{
      const t = (el.innerText || el.textContent || "").toLowerCase().trim();
      const kws = {BTN_KEYWORDS!r};
      return kws.some(k => t.includes(k));
    }};
    const results = [];
    const visit = (root) => {{
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
      let node;
      while ((node = walker.nextNode())) {{
        if (['BUTTON','A','DIV'].includes(node.tagName) && match(node)) {{
          results.push(node);
          break;
        }}
        if (node.shadowRoot) visit(node.shadowRoot);
      }}
    }};
    visit(document);
    if (results.length) {{ results[0].click(); return true; }}
    return false;
    """
    try:
        return bool(driver.execute_script(script))
    except Exception:
        return False


def _remove_overlays(driver):
    # Remove obvious overlays and unblock scrolling
    driver.execute_script("""
      (function(){
        const sels = [
          "[id*='cookie' i]", "[class*='cookie' i]",
          "[id*='consent' i]", "[class*='consent' i]",
          "[id*='cmp' i]", "[class*='cmp' i]",
          "[role='dialog'][aria-modal='true']"
        ];
        document.querySelectorAll(sels.join(',')).forEach(el => el.remove());
        // Nuke high z-index fixed overlays
        document.querySelectorAll('*').forEach(el => {
          const s = getComputedStyle(el);
          if (s.position==='fixed' && parseInt(s.zIndex||'0',10) > 999) el.remove();
        });
        document.documentElement.style.overflow = '';
        document.body.style.overflow = '';
      })();
    """)


def _handle_iframes(driver, handler):
    # Try iframes that likely contain CMPs
    frames = driver.find_elements(By.TAG_NAME, "iframe")
    for i, fr in enumerate(frames):
        src = (fr.get_attribute("src") or "").lower()
        if any(
            k in src for k in ["consent", "cookie", "cmp", "onetrust", "cookielaw", "didomi", "quantcast", "cookiebot"]
        ):
            try:
                driver.switch_to.frame(fr)
                if handler():  # call a local lambda that tries known clicks within this frame
                    driver.switch_to.default_content()
                    return True
            except Exception:
                pass
            driver.switch_to.default_content()
    return False


def kill_cookie_banners(driver, timeout=5):
    """
    Try multiple strategies in order. Returns True if it believes the banner was handled.
    Call AFTER driver.get(...), and be willing to call it again after lazy loads.
    """
    # 0) quick known selectors in main doc
    if _click_first(driver, KNOWN_SELECTORS):
        return True

    # 1) text-based click in main doc
    if _click_by_text(driver, timeout=timeout):
        return True

    # 2) try known selectors inside iframes
    if _handle_iframes(driver, lambda: _click_first(driver, KNOWN_SELECTORS) or _click_by_text(driver, timeout=2)):
        return True

    # 3) shadow DOM search/click
    if _click_in_shadow_dom(driver):
        return True

    # 4) as last resort, remove overlays
    _remove_overlays(driver)
    return False


def wait_for_page_content(driver, timeout=1):
    """
    Wait for main content to be loaded and visible.
    Tries <main>, <article>, then falls back to <body>.
    """
    try:
        # Try waiting for <main>
        WebDriverWait(driver, timeout).until(EC.visibility_of_element_located((By.TAG_NAME, "main")))
    except Exception:
        try:
            # Fallback: wait for <article>
            WebDriverWait(driver, timeout).until(EC.visibility_of_element_located((By.TAG_NAME, "article")))
        except Exception:
            # Fallback: wait for <body>
            WebDriverWait(driver, timeout).until(EC.visibility_of_element_located((By.TAG_NAME, "body")))
