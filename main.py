"""
Enhanced Playwright-based Instagram profile scraper
Features:
- Multiple proxy sources with health checking
- Advanced stealth techniques (canvas, webgl, fonts fingerprinting)
- Smart retry logic with circuit breaker
- Request/response caching
- Metrics and monitoring endpoints
- Dynamic user agent rotation
- Cookie management
- Enhanced error handling

Requirements:
    pip install fastapi uvicorn playwright httpx pydantic aiohttp redis
    playwright install chromium

Run:
    uvicorn enhanced_ig_scraper:app --host 0.0.0.0 --port 8000
"""
from typing import Optional, List, Dict, Set
import asyncio
import random
import time
import logging
import json
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from playwright.async_api import async_playwright, Browser, BrowserType, BrowserContext, Page

# -------------------- CONFIG --------------------
APP_NAME = "ENHANCED STEALTH SCRAPER v2.0"
POOL_SIZE = 8
CONCURRENCY = 12
RATE_LIMIT_PER_MIN = 80
PROXY_REFRESH_SECONDS = 45
PROXY_HEALTH_CHECK_INTERVAL = 120
CACHE_TTL_SECONDS = 300  # 5 minutes cache

# Multiple proxy sources for redundancy
PROXY_SOURCES = [
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all",
    "https://www.proxy-list.download/api/v1/get?type=http",
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
    "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/http.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
]

# Expanded user agent list
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.2; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
]

# Enhanced stealth script with fingerprint randomization
ADVANCED_STEALTH_SCRIPT = r"""
// Advanced stealth techniques
(() => {
    // Webdriver
    Object.defineProperty(navigator, 'webdriver', {get: () => false});
    
    // Chrome runtime
    window.navigator.chrome = {
        runtime: {},
        loadTimes: function() {},
        csi: function() {},
        app: {}
    };
    
    // Permissions
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) => (
        parameters.name === 'notifications'
            ? Promise.resolve({state: Notification.permission})
            : originalQuery(parameters)
    );
    
    // Plugins
    Object.defineProperty(navigator, 'plugins', {
        get: () => [1, 2, 3, 4, 5]
    });
    
    // Languages
    Object.defineProperty(navigator, 'languages', {
        get: () => ['en-US', 'en']
    });
    
    // Canvas fingerprint randomization
    const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
    HTMLCanvasElement.prototype.toDataURL = function(type) {
        if (type === 'image/png' && this.width === 0 && this.height === 0) {
            return originalToDataURL.apply(this, arguments);
        }
        const context = this.getContext('2d');
        const imageData = context.getImageData(0, 0, this.width, this.height);
        for (let i = 0; i < imageData.data.length; i += 4) {
            imageData.data[i] += Math.floor(Math.random() * 3) - 1;
        }
        context.putImageData(imageData, 0, 0);
        return originalToDataURL.apply(this, arguments);
    };
    
    // WebGL fingerprint randomization
    const getParameter = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(parameter) {
        if (parameter === 37445) {
            return 'Intel Inc.';
        }
        if (parameter === 37446) {
            return 'Intel Iris OpenGL Engine';
        }
        return getParameter.apply(this, [parameter]);
    };
    
    // Battery API
    if ('getBattery' in navigator) {
        navigator.getBattery = () => Promise.resolve({
            charging: true,
            chargingTime: 0,
            dischargingTime: Infinity,
            level: 1,
            addEventListener: () => {},
            removeEventListener: () => {}
        });
    }
    
    // Timezone consistency
    Date.prototype.getTimezoneOffset = function() { return -300; };
    
    // Media devices
    if (navigator.mediaDevices) {
        navigator.mediaDevices.enumerateDevices = () => Promise.resolve([
            {deviceId: 'default', kind: 'audioinput', label: '', groupId: ''},
            {deviceId: 'default', kind: 'audiooutput', label: '', groupId: ''},
            {deviceId: 'default', kind: 'videoinput', label: '', groupId: ''}
        ]);
    }
})();
"""

# -------------------- LOGGING --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("enhanced-scraper")

# -------------------- FASTAPI --------------------
app = FastAPI(title=APP_NAME, version="2.0")

# -------------------- METRICS --------------------
@dataclass
class Metrics:
    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    avg_response_time: float = 0.0
    active_proxies: int = 0
    proxy_failures: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recent_response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
metrics = Metrics()

# -------------------- RATE LIMITER --------------------
class TokenBucket:
    def __init__(self, rpm: int):
        self.capacity = rpm
        self.tokens = rpm
        self.fill_rate = rpm / 60.0
        self.ts = time.monotonic()
        self.lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.ts
            self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
            self.ts = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def wait_for_token(self):
        while True:
            ok = await self.consume(1)
            if ok:
                return
            await asyncio.sleep(0.05)

rate_bucket = TokenBucket(RATE_LIMIT_PER_MIN)
semaphore = asyncio.Semaphore(CONCURRENCY)

# -------------------- CACHE --------------------
_cache: Dict[str, tuple] = {}  # key -> (data, timestamp)
_cache_lock = asyncio.Lock()

async def get_cached(key: str) -> Optional[Dict]:
    async with _cache_lock:
        if key in _cache:
            data, ts = _cache[key]
            if time.time() - ts < CACHE_TTL_SECONDS:
                return data
            else:
                del _cache[key]
    return None

async def set_cache(key: str, data: Dict):
    async with _cache_lock:
        _cache[key] = (data, time.time())
        # cleanup old entries
        if len(_cache) > 1000:
            cutoff = time.time() - CACHE_TTL_SECONDS
            to_delete = [k for k, (_, ts) in _cache.items() if ts < cutoff]
            for k in to_delete:
                del _cache[k]

# -------------------- PROXY MANAGEMENT --------------------
@dataclass
class ProxyInfo:
    url: str
    failures: int = 0
    last_success: float = 0.0
    last_check: float = 0.0
    healthy: bool = True

_proxies: Dict[str, ProxyInfo] = {}
_proxy_lock = asyncio.Lock()
_circuit_breaker: Dict[str, int] = defaultdict(int)  # proxy -> failure count

async def fetch_proxies_from_source(source: str) -> List[str]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(source)
            if r.status_code != 200:
                return []
            lines = [l.strip() for l in r.text.splitlines()]
            proxies = []
            for line in lines:
                if ':' in line and not line.startswith('#'):
                    # handle various formats
                    if line.startswith('http://') or line.startswith('https://'):
                        proxies.append(line)
                    else:
                        proxies.append(f"http://{line}")
            return proxies
    except Exception as e:
        logger.warning(f"Failed to fetch from {source}: {e}")
        return []

async def fetch_all_proxies() -> List[str]:
    tasks = [fetch_proxies_from_source(src) for src in PROXY_SOURCES]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    all_proxies = []
    for result in results:
        if isinstance(result, list):
            all_proxies.extend(result)
    # deduplicate
    return list(set(all_proxies))

async def check_proxy_health(proxy_url: str) -> bool:
    """Quick health check for a proxy"""
    try:
        async with httpx.AsyncClient(proxies={"http://": proxy_url, "https://": proxy_url}, timeout=5.0) as client:
            r = await client.get("https://www.instagram.com", follow_redirects=True)
            return r.status_code in [200, 301, 302, 429]
    except Exception:
        return False

async def proxy_refresher():
    global _proxies
    while True:
        try:
            new_proxies = await fetch_all_proxies()
            if new_proxies:
                async with _proxy_lock:
                    # keep existing healthy proxies, add new ones
                    existing = set(_proxies.keys())
                    for proxy_url in new_proxies:
                        if proxy_url not in _proxies:
                            _proxies[proxy_url] = ProxyInfo(url=proxy_url)
                    
                    # remove proxies with too many failures
                    to_remove = [url for url, info in _proxies.items() 
                                if info.failures > 5 and not info.healthy]
                    for url in to_remove:
                        del _proxies[url]
                    
                    metrics.active_proxies = len([p for p in _proxies.values() if p.healthy])
                    logger.info(f"Proxy pool refreshed: {len(_proxies)} total, {metrics.active_proxies} healthy")
        except Exception as e:
            logger.error(f"Proxy refresher error: {e}")
        
        await asyncio.sleep(PROXY_REFRESH_SECONDS)

async def proxy_health_checker():
    """Background task to check proxy health"""
    while True:
        try:
            async with _proxy_lock:
                proxies_to_check = list(_proxies.items())
            
            # check a random subset
            sample_size = min(10, len(proxies_to_check))
            if proxies_to_check:
                sample = random.sample(proxies_to_check, sample_size)
                for url, info in sample:
                    if time.time() - info.last_check > 60:  # check at most once per minute
                        healthy = await check_proxy_health(url)
                        async with _proxy_lock:
                            if url in _proxies:
                                _proxies[url].healthy = healthy
                                _proxies[url].last_check = time.time()
                                if not healthy:
                                    _proxies[url].failures += 1
        except Exception as e:
            logger.error(f"Proxy health checker error: {e}")
        
        await asyncio.sleep(PROXY_HEALTH_CHECK_INTERVAL)

def choose_proxy() -> Optional[str]:
    if not _proxies:
        return None
    # filter healthy proxies with low failure rate
    healthy = [(url, info) for url, info in _proxies.items() 
               if info.healthy and info.failures < 3]
    if not healthy:
        # fallback to any proxy
        healthy = list(_proxies.items())
    if not healthy:
        return None
    
    # weighted random selection (prefer proxies with recent success)
    now = time.time()
    weights = []
    for url, info in healthy:
        recency = max(0, 300 - (now - info.last_success))  # decay over 5 min
        weight = 100 - info.failures * 10 + recency
        weights.append(max(1, weight))
    
    selected = random.choices(healthy, weights=weights, k=1)[0]
    return selected[0]

def report_proxy_result(proxy_url: Optional[str], success: bool):
    """Update proxy statistics"""
    if not proxy_url:
        return
    try:
        if proxy_url in _proxies:
            if success:
                _proxies[proxy_url].last_success = time.time()
                _proxies[proxy_url].failures = max(0, _proxies[proxy_url].failures - 1)
                _proxies[proxy_url].healthy = True
            else:
                _proxies[proxy_url].failures += 1
                if _proxies[proxy_url].failures > 3:
                    _proxies[proxy_url].healthy = False
                metrics.proxy_failures[proxy_url] += 1
    except Exception:
        pass

# -------------------- BROWSER POOL --------------------
class PooledBrowser:
    def __init__(self, browser: Browser, proxy: Optional[str], idx: int):
        self.browser = browser
        self.proxy = proxy
        self.idx = idx
        self.lock = asyncio.Lock()
        self.last_used = time.monotonic()
        self.use_count = 0

    async def close(self):
        try:
            await self.browser.close()
        except Exception:
            pass

class BrowserPool:
    def __init__(self, pool_size: int):
        self.pool_size = pool_size
        self.pool: List[PooledBrowser] = []
        self._init_lock = asyncio.Lock()
        self._type: Optional[BrowserType] = None

    async def init(self, p):
        async with self._init_lock:
            if self.pool:
                return
            self._type = p.chromium
            
            for i in range(self.pool_size):
                proxy = choose_proxy()
                logger.info(f"Launching browser {i} with proxy={proxy}")
                launch_args = {
                    "headless": True,
                    "args": [
                        "--no-sandbox",
                        "--disable-blink-features=AutomationControlled",
                        "--disable-dev-shm-usage",
                        "--disable-web-security",
                        "--disable-features=IsolateOrigins,site-per-process"
                    ]
                }
                if proxy:
                    launch_args["proxy"] = {"server": proxy}
                
                try:
                    browser = await self._type.launch(**launch_args)
                    pb = PooledBrowser(browser=browser, proxy=proxy, idx=i)
                    self.pool.append(pb)
                except Exception as e:
                    logger.error(f"Failed to launch browser {i}: {e}")
            
            logger.info(f"Browser pool initialized ({len(self.pool)} browsers)")

    async def get(self) -> PooledBrowser:
        while True:
            best = None
            for pb in self.pool:
                if not pb.lock.locked():
                    if best is None or pb.last_used < best.last_used:
                        best = pb
            if best:
                await best.lock.acquire()
                best.last_used = time.monotonic()
                best.use_count += 1
                return best
            await asyncio.sleep(0.01)

    async def release(self, pb: PooledBrowser):
        try:
            if pb.lock.locked():
                pb.lock.release()
        except RuntimeError:
            pass

    async def recycle_stale_browsers(self):
        """Recycle browsers that have been used too much or are stale"""
        for pb in self.pool:
            # recycle if used more than 50 times or idle for 10 minutes
            if pb.use_count > 50 or (time.monotonic() - pb.last_used > 600):
                if not pb.lock.locked():
                    try:
                        await pb.lock.acquire()
                        logger.info(f"Recycling browser {pb.idx} (uses: {pb.use_count})")
                        await pb.browser.close()
                        
                        proxy = choose_proxy()
                        launch_args = {
                            "headless": True,
                            "args": ["--no-sandbox", "--disable-blink-features=AutomationControlled"]
                        }
                        if proxy:
                            launch_args["proxy"] = {"server": proxy}
                        
                        new_browser = await self._type.launch(**launch_args)
                        pb.browser = new_browser
                        pb.proxy = proxy
                        pb.use_count = 0
                        pb.last_used = time.monotonic()
                    except Exception as e:
                        logger.error(f"Failed to recycle browser {pb.idx}: {e}")
                    finally:
                        try:
                            pb.lock.release()
                        except Exception:
                            pass

browser_pool = BrowserPool(POOL_SIZE)

# -------------------- SCRAPER LOGIC --------------------
RETRY_ATTEMPTS = 4
BASE_BACKOFF = 0.5

class ProfileResult(BaseModel):
    success: bool
    username: Optional[str] = None
    full_name: Optional[str] = None
    bio: Optional[str] = None
    followers: Optional[int] = None
    following: Optional[int] = None
    posts: Optional[int] = None
    verified: Optional[bool] = None
    private: Optional[bool] = None
    pic: Optional[str] = None
    external_url: Optional[str] = None
    is_business: Optional[bool] = None
    category: Optional[str] = None
    error: Optional[str] = None
    cached: bool = False
    meta: Optional[Dict] = None

async def parse_profile_data(html: str) -> Dict:
    """Enhanced parsing with multiple fallback strategies"""
    try:
        import re
        
        data = {}
        
        # Try to extract from various JSON structures
        patterns = {
            'followers': r'"edge_followed_by":\s*\{\s*"count":\s*(\d+)',
            'following': r'"edge_follow":\s*\{\s*"count":\s*(\d+)',
            'posts': r'"edge_owner_to_timeline_media":\s*\{\s*"count":\s*(\d+)',
            'full_name': r'"full_name":\s*"([^"]*)"',
            'bio': r'"biography":\s*"([^"]*)"',
            'pic': r'"profile_pic_url_hd":\s*"([^"]*)"',
            'external_url': r'"external_url":\s*"([^"]*)"',
            'category': r'"category_name":\s*"([^"]*)"',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, html)
            if match:
                value = match.group(1)
                if key in ['followers', 'following', 'posts']:
                    data[key] = int(value)
                else:
                    # unescape unicode
                    data[key] = value.encode().decode('unicode_escape')
        
        # Boolean fields
        data['verified'] = '"is_verified":true' in html
        data['private'] = '"is_private":true' in html
        data['is_business'] = '"is_business_account":true' in html
        
        return data
    except Exception as e:
        logger.exception(f"Parse error: {e}")
        return {}

async def scrape_profile(username: str, screenshot: bool = False, use_cache: bool = True) -> ProfileResult:
    start_time = time.time()
    username = username.lstrip('@').lower().strip()
    
    # check cache
    cache_key = f"profile:{username}"
    if use_cache:
        cached = await get_cached(cache_key)
        if cached:
            logger.info(f"Cache hit for {username}")
            cached['cached'] = True
            return ProfileResult(**cached)
    
    # rate limiting
    await rate_bucket.wait_for_token()
    
    async with semaphore:
        pb = None
        try:
            pb = await browser_pool.get()
            
            # create context with enhanced stealth
            ua = random.choice(USER_AGENTS)
            viewport = {
                "width": random.randint(1024, 1920),
                "height": random.randint(768, 1080)
            }
            
            context: BrowserContext = await pb.browser.new_context(
                user_agent=ua,
                viewport=viewport,
                locale='en-US',
                timezone_id='America/New_York',
                permissions=['geolocation'],
                geolocation={'latitude': 40.7128, 'longitude': -74.0060},
                color_scheme='light'
            )
            
            await context.add_init_script(ADVANCED_STEALTH_SCRIPT)
            page: Page = await context.new_page()
            
            # set extra headers
            await page.set_extra_http_headers({
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            })
            
            attempt = 0
            last_err = None
            
            while attempt < RETRY_ATTEMPTS:
                attempt += 1
                try:
                    url = f"https://www.instagram.com/{username}/"
                    
                    # navigate with wait
                    await page.goto(url, wait_until='domcontentloaded', timeout=20000)
                    
                    # random human-like delay
                    await asyncio.sleep(random.uniform(0.5, 1.5))
                    
                    # wait for content
                    try:
                        await page.wait_for_selector('header, main, h1', timeout=10000)
                    except Exception:
                        pass
                    
                    # scroll to trigger lazy loading
                    await page.evaluate('window.scrollBy(0, 300)')
                    await asyncio.sleep(0.3)
                    
                    html = await page.content()
                    
                    # check for rate limit or block
                    if 'Please wait a few minutes before you try again' in html:
                        raise Exception("Rate limited by Instagram")
                    
                    if 'Sorry, this page' in html or 'Page Not Found' in html:
                        raise Exception("Profile not found")
                    
                    parsed = await parse_profile_data(html)
                    
                    elapsed = time.time() - start_time
                    metrics.recent_response_times.append(elapsed)
                    
                    meta = {
                        'proxy': pb.proxy,
                        'browser_idx': pb.idx,
                        'ua': ua[:50],
                        'attempts': attempt,
                        'response_time': round(elapsed, 2),
                        'timestamp': datetime.utcnow().isoformat() + 'Z'
                    }
                    
                    result_data = {
                        'success': True,
                        'username': username,
                        'full_name': parsed.get('full_name'),
                        'bio': parsed.get('bio'),
                        'followers': parsed.get('followers'),
                        'following': parsed.get('following'),
                        'posts': parsed.get('posts'),
                        'verified': parsed.get('verified'),
                        'private': parsed.get('private'),
                        'pic': parsed.get('pic'),
                        'external_url': parsed.get('external_url'),
                        'is_business': parsed.get('is_business'),
                        'category': parsed.get('category'),
                        'meta': meta
                    }
                    
                    # optional screenshot
                    if screenshot:
                        try:
                            img_bytes = await page.screenshot(full_page=False, type='jpeg', quality=60)
                            meta['screenshot_size'] = len(img_bytes)
                        except Exception:
                            pass
                    
                    await context.close()
                    
                    # cache result
                    await set_cache(cache_key, result_data)
                    
                    # update metrics
                    metrics.requests_success += 1
                    report_proxy_result(pb.proxy, True)
                    
                    return ProfileResult(**result_data)
                    
                except Exception as e:
                    last_err = str(e)
                    logger.warning(f"Attempt {attempt} for {username} failed: {e}")
                    report_proxy_result(pb.proxy, False)
                    
                    if attempt < RETRY_ATTEMPTS:
                        backoff = BASE_BACKOFF * (2 ** (attempt - 1)) + random.random()
                        await asyncio.sleep(backoff)
            
            # all retries failed
            await context.close()
            metrics.requests_failed += 1
            
            return ProfileResult(
                success=False,
                username=username,
                error=last_err or "Max retries exceeded",
                meta={'proxy': pb.proxy, 'attempts': attempt}
            )
            
        except Exception as e:
            logger.error(f"Critical error scraping {username}: {e}")
            metrics.requests_failed += 1
            return ProfileResult(
                success=False,
                username=username,
                error=str(e),
                meta={'proxy': pb.proxy if pb else None}
            )
        finally:
            if pb:
                await browser_pool.release(pb)
            metrics.requests_total += 1
            
            # update avg response time
            if metrics.recent_response_times:
                metrics.avg_response_time = sum(metrics.recent_response_times) / len(metrics.recent_response_times)

# -------------------- STARTUP & BACKGROUND --------------------
@app.on_event("startup")
async def startup():
    logger.info("Starting up...")
    
    # start background tasks
    asyncio.create_task(proxy_refresher())
    asyncio.create_task(proxy_health_checker())
    
    # initial proxy fetch
    logger.info("Fetching initial proxy pool...")
    proxies = await fetch_all_proxies()
    async with _proxy_lock:
        for proxy_url in proxies:
            _proxies[proxy_url] = ProxyInfo(url=proxy_url)
    logger.info(f"Initial proxy pool: {len(_proxies)} proxies")
    
    # initialize playwright
    p = await async_playwright().start()
    await browser_pool.init(p)
    
    # start browser recycler
    asyncio.create_task(browser_recycler_loop())
    
    logger.info(f"Startup complete - {len(browser_pool.pool)} browsers ready")

async def browser_recycler_loop():
    while True:
        try:
            await browser_pool.recycle_stale_browsers()
        except Exception as e:
            logger.error(f"Browser recycler error: {e}")
        await asyncio.sleep(180)  # every 3 minutes

# -------------------- ENDPOINTS --------------------
@app.get("/")
async def home():
    return {
        "service": APP_NAME,
        "status": "online",
        "endpoints": {
            "/ig": "Scrape single profile",
            "/bulk": "Scrape multiple profiles",
            "/metrics": "Service metrics",
            "/health": "Health check",
            "/cache/clear": "Clear cache"
        },
        "docs": "/docs"
    }

@app.get("/ig")
async def ig(
    user: str = Query(..., description="Instagram username"),
    screenshot: bool = Query(False, description="Capture screenshot"),
    nocache: bool = Query(False, description="Skip cache")
):
    if not user:
        raise HTTPException(status_code=400, detail="user parameter required")
    
    result = await scrape_profile(user, screenshot=screenshot, use_cache=not nocache)
    return JSONResponse(content=result.dict())

@app.get("/bulk")
async def bulk(
    users: str = Query(..., description="Comma-separated usernames"),
    screenshot: bool = Query(False, description="Capture screenshots"),
    nocache: bool = Query(False, description="Skip cache")
):
    if not users:
        raise HTTPException(status_code=400, detail="users parameter required")
    
    usernames = [u.strip() for u in users.split(',') if u.strip()]
    if len(usernames) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 users per bulk request")
    
    tasks = [scrape_profile(u, screenshot=screenshot, use_cache=not nocache) for u in usernames]
    results = await asyncio.gather(*tasks)
    
    success_count = sum(1 for r in results if r.success)
    
    return {
        "total": len(results),
        "successful": success_count,
        "failed": len(results) - success_count,
        "results": [r.dict() for r in results]
    }

@app.get("/metrics")
async def get_metrics():
    """Get service performance metrics"""
    async with _proxy_lock:
        proxy_stats = {
            "total": len(_proxies),
            "healthy": len([p for p in _proxies.values() if p.healthy]),
            "unhealthy": len([p for p in _proxies.values() if not p.healthy]),
            "top_failures": dict(sorted(
                metrics.proxy_failures.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])
        }
    
    return {
        "service": APP_NAME,
        "uptime_seconds": int(time.time() - _start_time),
        "requests": {
            "total": metrics.requests_total,
            "successful": metrics.requests_success,
            "failed": metrics.requests_failed,
            "success_rate": round(
                metrics.requests_success / metrics.requests_total * 100, 2
            ) if metrics.requests_total > 0 else 0
        },
        "performance": {
            "avg_response_time": round(metrics.avg_response_time, 2),
            "recent_samples": len(metrics.recent_response_times)
        },
        "proxies": proxy_stats,
        "pool": {
            "size": len(browser_pool.pool),
            "browsers": [
                {
                    "idx": pb.idx,
                    "proxy": pb.proxy,
                    "uses": pb.use_count,
                    "locked": pb.lock.locked()
                }
                for pb in browser_pool.pool
            ]
        },
        "cache": {
            "entries": len(_cache),
            "ttl_seconds": CACHE_TTL_SECONDS
        },
        "rate_limit": {
            "rpm": RATE_LIMIT_PER_MIN,
            "concurrency": CONCURRENCY,
            "available_tokens": int(rate_bucket.tokens)
        }
    }

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    async with _proxy_lock:
        healthy_proxies = len([p for p in _proxies.values() if p.healthy])
    
    is_healthy = (
        len(browser_pool.pool) > 0 and
        healthy_proxies > 0
    )
    
    return {
        "status": "healthy" if is_healthy else "degraded",
        "browsers": len(browser_pool.pool),
        "healthy_proxies": healthy_proxies,
        "timestamp": datetime.utcnow().isoformat() + 'Z'
    }

@app.post("/cache/clear")
async def clear_cache():
    """Clear the result cache"""
    async with _cache_lock:
        count = len(_cache)
        _cache.clear()
    
    return {
        "message": "Cache cleared",
        "entries_removed": count
    }

@app.get("/proxies")
async def list_proxies(healthy_only: bool = Query(False)):
    """List current proxy pool status"""
    async with _proxy_lock:
        proxy_list = []
        for url, info in _proxies.items():
            if healthy_only and not info.healthy:
                continue
            proxy_list.append({
                "url": url,
                "healthy": info.healthy,
                "failures": info.failures,
                "last_success": info.last_success,
                "last_check": info.last_check
            })
    
    return {
        "total": len(proxy_list),
        "proxies": sorted(proxy_list, key=lambda x: x['failures'])
    }

@app.post("/proxies/refresh")
async def force_proxy_refresh(background_tasks: BackgroundTasks):
    """Force an immediate proxy refresh"""
    background_tasks.add_task(fetch_and_update_proxies)
    return {"message": "Proxy refresh initiated"}

async def fetch_and_update_proxies():
    """Helper for background proxy refresh"""
    try:
        new_proxies = await fetch_all_proxies()
        if new_proxies:
            async with _proxy_lock:
                for proxy_url in new_proxies:
                    if proxy_url not in _proxies:
                        _proxies[proxy_url] = ProxyInfo(url=proxy_url)
            logger.info(f"Manual proxy refresh: {len(new_proxies)} proxies fetched")
    except Exception as e:
        logger.error(f"Manual proxy refresh failed: {e}")

@app.get("/test")
async def test_scrape():
    """Test endpoint with a known working profile"""
    result = await scrape_profile("instagram", screenshot=False, use_cache=False)
    return result.dict()

# -------------------- CLEANUP --------------------
@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down - closing browsers")
    try:
        for pb in browser_pool.pool:
            await pb.close()
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# -------------------- GLOBAL STATE --------------------
_start_time = time.time()

# -------------------- ADDITIONAL UTILITIES --------------------

async def get_profile_batch(usernames: List[str], batch_size: int = 5) -> List[ProfileResult]:
    """
    Process profiles in smaller batches to avoid overwhelming the system
    """
    results = []
    for i in range(0, len(usernames), batch_size):
        batch = usernames[i:i + batch_size]
        batch_results = await asyncio.gather(*[scrape_profile(u) for u in batch])
        results.extend(batch_results)
        
        # small delay between batches
        if i + batch_size < len(usernames):
            await asyncio.sleep(1)
    
    return results

@app.get("/batch")
async def batch_scrape(
    users: str = Query(..., description="Comma-separated usernames"),
    batch_size: int = Query(5, description="Batch size", ge=1, le=10)
):
    """
    Scrape profiles in controlled batches
    Useful for large lists to avoid rate limiting
    """
    usernames = [u.strip() for u in users.split(',') if u.strip()]
    
    if len(usernames) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 users for batch endpoint")
    
    results = await get_profile_batch(usernames, batch_size)
    
    success_count = sum(1 for r in results if r.success)
    
    return {
        "total": len(results),
        "successful": success_count,
        "failed": len(results) - success_count,
        "batch_size": batch_size,
        "results": [r.dict() for r in results]
    }

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "pool_size": POOL_SIZE,
        "concurrency": CONCURRENCY,
        "rate_limit_rpm": RATE_LIMIT_PER_MIN,
        "proxy_refresh_seconds": PROXY_REFRESH_SECONDS,
        "proxy_sources": len(PROXY_SOURCES),
        "user_agents": len(USER_AGENTS),
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "retry_attempts": RETRY_ATTEMPTS
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
