// Global variables
let movieData = [];
let filteredData = [];
// Live comparison state
const liveCompare = {
    items: [], // {movie_name, prediction, confidence, social_data, key_factors}
    byName: new Map()
};

// Persist live compare between reloads
try {
    const saved = JSON.parse(localStorage.getItem('liveCompareItems') || '[]');
    if (Array.isArray(saved)) {
        liveCompare.items = saved;
        liveCompare.byName = new Map(saved.map(x => [x.movie_name, x]));
    }
} catch { }

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function () {
    initializeDashboard();
});

async function initializeDashboard() {
    showLoading();

    try {
        await loadData();
        setupEventListeners();
    // Render any persisted live compare state
    renderLiveCompareChips();
    updateLiveComparisonCharts();
        updateOverviewMetrics();
        createSuccessDistributionChart();
        createEngagementChart();
        populateMovieSelector();
        generateInsights();
        hideLoading();
    } catch (error) {
        console.error('Error initializing dashboard:', error);
        hideLoading();
        showError('Failed to load data. Please check if pre_release_movie_dataset.json exists.');
    }
}

// Load data from JSON file
async function loadData() {
    try {
        // Try to load real movie data first
        let response = await fetch('real_movie_dataset.json');
        if (!response.ok) {
            // Fallback to synthetic data
            console.log('Real data not found, trying synthetic data...');
            response = await fetch('pre_release_movie_dataset.json');
        }

        if (!response.ok) {
            throw new Error('Failed to fetch data');
        }

        movieData = await response.json();
        filteredData = [...movieData];
        console.log('Data loaded:', movieData.length, 'records');
        console.log('Sample record:', movieData[0]);

        // Display data source info
        const dataSource = response.url.includes('real_movie') ? 'Real Movie Data (Based on TMDB)' : 'Synthetic Data';
        displayDataSourceInfo(dataSource);

    } catch (error) {
        console.error('Error loading data:', error);
        // Create sample data for demo if file not found
        createSampleData();
    }
}

// Create sample data for demo
function createSampleData() {
    const movies = ['Avengers: Endgame', 'Avatar 2', 'Top Gun: Maverick', 'Black Panther 2'];
    const genres = ['Action', 'Sci-Fi', 'Drama'];
    const buzzTypes = ['Social Media', 'News', 'Trailer', 'Reviews'];
    const successLabels = ['Hit', 'Average', 'Flop'];

    movieData = [];

    for (let i = 0; i < 500; i++) {
        const movie = movies[Math.floor(Math.random() * movies.length)];
        const genre = genres[Math.floor(Math.random() * genres.length)];
        const buzzType = buzzTypes[Math.floor(Math.random() * buzzTypes.length)];
        const label = successLabels[Math.floor(Math.random() * successLabels.length)];
        const daysBeforeRelease = Math.floor(Math.random() * 365) + 1;

        const baseEngagement = label === 'Hit' ? 0.7 : label === 'Average' ? 0.4 : 0.2;
        const engagement = baseEngagement + (Math.random() - 0.5) * 0.3;

        movieData.push({
            movie_name: movie,
            genre: genre,
            buzz_type: buzzType,
            label: label,
            days_before_release: daysBeforeRelease,
            likes: Math.floor(Math.random() * 10000),
            shares: Math.floor(Math.random() * 5000),
            comments: Math.floor(Math.random() * 2000),
            post_count: Math.floor(Math.random() * 100) + 1,
            engagement_rate: Math.max(0, Math.min(1, engagement))
        });
    }

    filteredData = [...movieData];
    console.log('Sample data created:', movieData.length, 'records');
}

// Setup event listeners
function setupEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            const section = this.dataset.section;
            showSection(section);
        });
    });

    // Filters
    document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
        checkbox.addEventListener('change', applyFilters);
    });

    document.getElementById('time-range').addEventListener('input', function () {
        document.getElementById('time-range-value').textContent = this.value + ' days';
        applyFilters();
    });

    // Prediction form
    document.getElementById('prediction-form').addEventListener('submit', handlePrediction);

    // Range inputs
    document.getElementById('social-buzz').addEventListener('input', function () {
        document.getElementById('social-buzz-value').textContent = this.value;
    });

    document.getElementById('engagement-rate-input').addEventListener('input', function () {
        document.getElementById('engagement-rate-value').textContent = parseFloat(this.value).toFixed(2);
    });

    // Live prediction form
    const liveForm = document.getElementById('live-prediction-form');
    if (liveForm) liveForm.addEventListener('submit', handleLivePredictionSubmit);

    // Compare actions
    const addBtn = document.getElementById('add-to-compare');
    if (addBtn) addBtn.addEventListener('click', addCurrentLiveToCompare);
    const clearBtn = document.getElementById('clear-compare');
    if (clearBtn) clearBtn.addEventListener('click', clearLiveCompare);
}

// Show section
function showSection(sectionName) {
    // Update navigation
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[data-section="${sectionName}"]`).classList.add('active');

    // Show section
    document.querySelectorAll('.section').forEach(section => section.classList.remove('active'));
    document.getElementById(sectionName).classList.add('active');

    // Load section-specific content
    switch (sectionName) {
        case 'timeline':
            createTimelineChart();
            createBuzzTypeChart();
            createEngagementTimelineChart();
            createCumulativeBuzzChart();
            break;
        case 'comparison':
            updateMovieComparison();
            break;
        case 'real-data':
            createRealDataAnalysis();
            break;
        case 'insights':
            generateInsights();
            break;
    }
}

// ==== Live prediction integration ====
async function handleLivePredictionSubmit(e) {
    e.preventDefault();
    const titleEl = document.getElementById('live-movie-title');
    const title = (titleEl?.value || '').trim();
    if (!title) return;

    const panel = document.getElementById('live-prediction-results');
    const loading = document.getElementById('live-loading');
    const content = document.getElementById('live-result-content');
    panel.style.display = 'block';
    loading.style.display = 'flex';
    content.style.display = 'none';

    try {
        const res = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ movie_name: title })
        });

        const data = await res.json();
        if (!res.ok || !data || data.success === false) throw new Error(data?.error || 'Prediction failed');

        showLivePrediction(data);
        cacheCurrentLive(data);
        await refreshRecentPredictions();
    } catch (err) {
        console.warn('Live API failed, using fallback demo:', err.message);
        const demo = generateDemoLiveResult(title);
        showLivePrediction(demo);
        cacheCurrentLive(demo);
    } finally {
        loading.style.display = 'none';
        content.style.display = 'block';
    }
}

function showLivePrediction(result) {
    const cat = document.getElementById('live-prediction-category');
    const conf = document.getElementById('live-confidence-score');
    const ts = document.getElementById('live-data-timestamp');
    const tm = document.getElementById('live-total-mentions');
    const ss = document.getElementById('live-sentiment-score');
    const pr = document.getElementById('live-positive-ratio');
    const vp = document.getElementById('live-viral-potential');
    const factors = document.getElementById('live-key-factors');

    cat.textContent = result.prediction;
    cat.className = `live-prediction-badge ${result.prediction.toLowerCase()}`;
    conf.textContent = `${result.confidence}%`;
    ts.textContent = result.timestamp ? new Date(result.timestamp).toLocaleString() : 'Real-time data';

    tm.textContent = result.social_data?.total_mentions ?? '-';
    ss.textContent = (result.social_data?.sentiment_score ?? 0).toFixed(3);
    pr.textContent = ((result.social_data?.positive_ratio ?? 0) * 100).toFixed(1) + '%';
    vp.textContent = ((result.social_data?.viral_potential ?? 0) * 100).toFixed(1) + '%';

    factors.innerHTML = '';
    (result.key_factors || []).forEach(f => {
        const li = document.createElement('li');
        li.textContent = f;
        factors.appendChild(li);
    });
}

function cacheCurrentLive(result) {
    if (!result?.movie_name) return;
    liveCompare.byName.set(result.movie_name, result);
    const exists = liveCompare.items.findIndex(x => x.movie_name === result.movie_name);
    if (exists >= 0) liveCompare.items[exists] = result; else liveCompare.items.push(result);
    try { localStorage.setItem('liveCompareItems', JSON.stringify(liveCompare.items)); } catch { }
    renderLiveCompareChips();
    updateLiveComparisonCharts();
}

function addCurrentLiveToCompare() {
    const title = (document.getElementById('live-movie-title')?.value || '').trim();
    if (!title) return;
    // Already cached during show; just ensure charts refresh
    updateLiveComparisonCharts();
}

function clearLiveCompare() {
    liveCompare.items = [];
    liveCompare.byName.clear();
    try { localStorage.removeItem('liveCompareItems'); } catch { }
    renderLiveCompareChips();
    updateLiveComparisonCharts();
}

function renderLiveCompareChips() {
    const list = document.getElementById('live-compare-list');
    if (!list) return;
    list.innerHTML = '';
    liveCompare.items.forEach(item => {
        const chip = document.createElement('div');
        chip.className = 'chip';
        chip.style.cssText = 'padding:6px 10px;border-radius:16px;background:#ecf0f1;display:flex;align-items:center;gap:6px;';
        chip.innerHTML = `<span><strong>${item.movie_name}</strong> • ${item.prediction} • ${item.confidence}%</span>`;
        const x = document.createElement('button');
        x.textContent = '×';
        x.title = 'Remove';
        x.style.cssText = 'border:none;background:transparent;font-size:16px;cursor:pointer;';
        x.onclick = () => {
            liveCompare.items = liveCompare.items.filter(i => i.movie_name !== item.movie_name);
            liveCompare.byName.delete(item.movie_name);
            renderLiveCompareChips();
            updateLiveComparisonCharts();
        };
        chip.appendChild(x);
        list.appendChild(chip);
    });
}

function updateLiveComparisonCharts() {
    // Radar chart: sentiment score, positive ratio, engagement rate, viral potential, mention velocity (scaled)
    const metrics = ['sentiment', 'positive', 'engagement', 'viral', 'velocity'];
    const thetaLabels = ['Sentiment', 'Positive', 'Engagement', 'Viral', 'Velocity'];

    const radarTraces = liveCompare.items.map((item, idx) => {
        const s = item.social_data || {};
        const r = [
            normalize(s.sentiment_score, -1, 1),
            s.positive_ratio ?? 0,
            s.engagement_rate ?? 0,
            s.viral_potential ?? 0,
            normalize(s.mention_velocity ?? 0, 0, 20)
        ];
        return {
            type: 'scatterpolar',
            r,
            theta: thetaLabels,
            fill: 'toself',
            name: item.movie_name
        };
    });

    Plotly.newPlot('live-radar-chart', radarTraces, {
        polar: { radialaxis: { visible: true, range: [0, 1] } },
        margin: { t: 20, b: 20, l: 20, r: 20 }
    }, { responsive: true });

    // Confidence bars
    const confTrace = {
        x: liveCompare.items.map(i => i.movie_name),
        y: liveCompare.items.map(i => i.confidence || 0),
        type: 'bar',
        marker: { color: liveCompare.items.map(i => colorByPrediction(i.prediction)) },
        text: liveCompare.items.map(i => (i.prediction_score != null ? `Score ${i.prediction_score}` : '')),
        textposition: 'auto'
    };
    Plotly.newPlot('live-confidence-chart', [confTrace], {
        yaxis: { title: 'Confidence (%)', range: [0, 100] },
        margin: { t: 20, b: 60, l: 50, r: 20 }
    }, { responsive: true });

    // Sentiment stacked
    const sentTraces = ['positive_ratio', 'negative_ratio'].map((k, idx) => ({
        x: liveCompare.items.map(i => i.movie_name),
        y: liveCompare.items.map(i => (i.social_data?.[k] || 0) * 100),
        type: 'bar',
        name: k === 'positive_ratio' ? 'Positive %' : 'Negative %',
        marker: { color: k === 'positive_ratio' ? '#27ae60' : '#e74c3c' }
    }));
    Plotly.newPlot('live-sentiment-stacked', sentTraces, {
        barmode: 'stack',
        yaxis: { title: 'Percent (%)', range: [0, 100] },
        margin: { t: 20, b: 60, l: 50, r: 20 }
    }, { responsive: true });

    // Buzz funnel (Mentions -> Positives -> Engaged -> Viral)
    buildLiveFunnelChart();

    // Average confidence gauge
    buildLiveGauge();

    // Comparison table
    renderCompareTable();

    // Score vs confidence scatter
    buildLiveScoreScatter();

    // Mentions vs Engagement bubble
    buildLiveBubble();

    // Velocity vs Viral Potential scatter
    buildVelocityViral();
}

// Seeded random based on string
function seededRandom(str) {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 16777619);
    }
    // LCG
    let seed = h >>> 0;
    return () => {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        return (seed & 0xfffffff) / 0xfffffff;
    };
}

function generateDemoLiveResult(title) {
    const rnd = seededRandom(title);
    const mentions = Math.floor(80 + rnd() * 1200);
    const sentiment = -0.2 + rnd() * 0.7; // -0.2..0.5
    const positive = Math.min(0.9, Math.max(0.05, 0.4 + sentiment * 0.6 + (rnd() - 0.5) * 0.1));
    const negative = Math.min(0.9, Math.max(0.05, 0.35 - sentiment * 0.5 + (rnd() - 0.5) * 0.1));
    const engagement = 0.08 + rnd() * 0.7; // 0.08..0.78
    const viral = 0.03 + rnd() * 0.25; // 0.03..0.28
    const velocity = 1 + rnd() * 30; // 1..30

    // Recreate backend-like score
    const weights = { sentiment: 0.35, engagement: 0.25, volume: 0.20, viral: 0.15, demo: 0.05 };
    const score = (
        sentiment * weights.sentiment +
        Math.min(engagement / 1.0, 1.0) * weights.engagement +
        Math.min(velocity / 50, 1.0) * weights.volume +
        viral * weights.viral +
        Math.min(positive, 1.0) * weights.demo
    );
    const normalized = Math.max(0, Math.min(1, score + 0.5));
    let prediction, confidence;
    if (normalized >= 0.7) {
        prediction = 'Hit';
        confidence = Math.min(95, 70 + (normalized - 0.7) * 83);
    } else if (normalized >= 0.4) {
        prediction = 'Average';
        confidence = Math.min(85, 50 + (normalized - 0.4) * 116);
    } else {
        prediction = 'Flop';
        confidence = Math.min(75, 30 + normalized * 67);
    }

    return {
        movie_name: title,
        prediction,
        confidence: Math.round(confidence * 10) / 10,
        prediction_score: Math.round(normalized * 1000) / 1000,
        social_data: {
            total_mentions: mentions,
            sentiment_score: Math.round(sentiment * 1000) / 1000,
            positive_ratio: Math.round(positive * 1000) / 1000,
            negative_ratio: Math.round(negative * 1000) / 1000,
            engagement_rate: Math.round(engagement * 1000) / 1000,
            viral_potential: Math.round(viral * 1000) / 1000,
            mention_velocity: Math.round(velocity * 10) / 10
        },
        key_factors: [
            `Sentiment Score: ${sentiment.toFixed(3)}`,
            `Positive Mentions: ${(positive * 100).toFixed(1)}%`,
            `Engagement Rate: ${engagement.toFixed(2)}`,
            `Viral Potential: ${(viral * 100).toFixed(1)}%`,
            `Mention Velocity: ${velocity.toFixed(1)}/hour`
        ],
        timestamp: new Date().toISOString(),
        data_freshness: 'Simulated'
    };
}

function buildLiveBubble() {
    const el = document.getElementById('live-bubble');
    if (!el) return;
    const x = liveCompare.items.map(i => i.social_data?.total_mentions || 0);
    const y = liveCompare.items.map(i => (i.social_data?.engagement_rate || 0));
    const size = liveCompare.items.map(i => 10 + ((i.confidence || 0) / 100) * 30);
    const color = liveCompare.items.map(i => colorByPrediction(i.prediction));
    const text = liveCompare.items.map(i => i.movie_name);
    Plotly.newPlot('live-bubble', [{ x, y, text, mode: 'markers', type: 'scatter', marker: { size, color, sizemode: 'diameter', opacity: 0.8 } }], {
        xaxis: { title: 'Mentions' }, yaxis: { title: 'Engagement Rate' }, margin: { t: 20, b: 60, l: 60, r: 20 }
    }, { responsive: true });
}

function buildVelocityViral() {
    const el = document.getElementById('live-velocity-viral');
    if (!el) return;
    const x = liveCompare.items.map(i => i.social_data?.mention_velocity || 0);
    const y = liveCompare.items.map(i => (i.social_data?.viral_potential || 0) * 100);
    const text = liveCompare.items.map(i => i.movie_name);
    const color = liveCompare.items.map(i => colorByPrediction(i.prediction));
    Plotly.newPlot('live-velocity-viral', [{ x, y, text, mode: 'markers', type: 'scatter', marker: { size: 12, color } }], {
        xaxis: { title: 'Mention Velocity (per hour)' }, yaxis: { title: 'Viral Potential (%)', range: [0, 100] }, margin: { t: 20, b: 60, l: 60, r: 20 }
    }, { responsive: true });
}

function buildLiveFunnelChart() {
    const sum = (arr, sel) => arr.reduce((s, x) => s + (sel(x) || 0), 0);
    const totalMentions = sum(liveCompare.items, x => x.social_data?.total_mentions);
    const positives = sum(liveCompare.items, x => (x.social_data?.total_mentions || 0) * (x.social_data?.positive_ratio || 0));
    const engaged = sum(liveCompare.items, x => (x.social_data?.total_mentions || 0) * (x.social_data?.engagement_rate || 0));
    const viral = sum(liveCompare.items, x => (x.social_data?.total_mentions || 0) * (x.social_data?.viral_potential || 0));

    const values = [totalMentions, positives, engaged, viral].map(v => Math.round(v));
    const labels = ['Mentions', 'Positive Mentions', 'Engaged Users', 'Viral Reach'];

    if (document.getElementById('live-funnel-chart')) {
        const trace = {
            type: 'funnel',
            y: labels,
            x: values,
            textinfo: 'value+percent initial',
            marker: { color: ['#7f8c8d', '#27ae60', '#3498db', '#8e44ad'] }
        };
        Plotly.newPlot('live-funnel-chart', [trace], { margin: { t: 10, b: 10, l: 50, r: 20 } }, { responsive: true });
    }
}

function buildLiveGauge() {
    if (!document.getElementById('live-gauge-chart')) return;
    const avg = liveCompare.items.length
        ? liveCompare.items.reduce((s, x) => s + (x.confidence || 0), 0) / liveCompare.items.length
        : 0;
    const trace = {
        domain: { x: [0, 1], y: [0, 1] },
        value: avg,
        title: { text: 'Avg Confidence (%)' },
        type: 'indicator',
        mode: 'gauge+number',
        gauge: {
            axis: { range: [0, 100] },
            bar: { color: '#2980b9' },
            steps: [
                { range: [0, 40], color: '#f8d7da' },
                { range: [40, 70], color: '#fff3cd' },
                { range: [70, 100], color: '#d4edda' }
            ]
        }
    };
    Plotly.newPlot('live-gauge-chart', [trace], { margin: { t: 10, b: 10, l: 20, r: 20 } }, { responsive: true });
}

function renderCompareTable() {
    const el = document.getElementById('live-compare-table');
    if (!el) return;
    if (liveCompare.items.length === 0) { el.innerHTML = '<p>No items yet. Add a few live predictions.</p>'; return; }
    const rows = liveCompare.items.map(x => {
        const s = x.social_data || {};
        return `<tr>
            <td>${x.movie_name}</td>
            <td><span class="success-badge ${x.prediction.toLowerCase()}">${x.prediction}</span></td>
            <td>${x.confidence ?? '-'}</td>
            <td>${s.total_mentions ?? '-'}</td>
            <td>${(s.sentiment_score ?? 0).toFixed(3)}</td>
            <td>${((s.positive_ratio ?? 0) * 100).toFixed(0)}%</td>
            <td>${((s.engagement_rate ?? 0) * 100).toFixed(0)}%</td>
            <td>${((s.viral_potential ?? 0) * 100).toFixed(0)}%</td>
        </tr>`;
    }).join('');
    el.innerHTML = `
        <table class="movie-table">
            <thead>
                <tr>
                    <th>Movie</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                    <th>Mentions</th>
                    <th>Sentiment</th>
                    <th>Positive%</th>
                    <th>Engagement%</th>
                    <th>Viral%</th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>`;
}

function colorByPrediction(p) {
    if (p === 'Hit') return '#27ae60';
    if (p === 'Average') return '#f39c12';
    return '#e74c3c';
}

function normalize(v, min, max) {
    if (v == null) return 0;
    if (max === min) return 0;
    const n = (v - min) / (max - min);
    return Math.max(0, Math.min(1, n));
}

// Recent predictions panel
async function refreshRecentPredictions() {
    try {
        const res = await fetch('http://localhost:5000/recent-predictions');
        const data = await res.json();
        if (!res.ok || data.success === false) return;
        renderRecentPredictions(data.data || []);
    } catch { }
}

function renderRecentPredictions(items) {
    const wrap = document.getElementById('recent-predictions');
    if (!wrap) return;
    wrap.innerHTML = '';
    items.forEach(x => {
        const card = document.createElement('div');
        card.className = 'result-card';
        card.innerHTML = `
            <div class="result-main">
                <h4>${x.movie_name}</h4>
                <div class="prediction-badge ${x.prediction.toLowerCase()}">${x.prediction}</div>
                <div class="confidence-score"><span>Confidence:</span> <span>${x.confidence}%</span></div>
            </div>
            <p style="opacity:.7;margin:6px 0 0;">${new Date(x.timestamp).toLocaleString()}</p>
        `;
        wrap.appendChild(card);
    });
}

// Apply filters
function applyFilters() {
    const checkedCategories = Array.from(document.querySelectorAll('input[type="checkbox"]:checked'))
        .map(cb => cb.value);
    const maxDays = parseInt(document.getElementById('time-range').value);

    filteredData = movieData.filter(item =>
        checkedCategories.includes(item.label) &&
        item.days_before_release <= maxDays
    );

    // Update charts based on current section
    const activeSection = document.querySelector('.section.active').id;
    if (activeSection === 'timeline') {
        createTimelineChart();
        createBuzzTypeChart();
        createEngagementTimelineChart();
    }
}

// Update overview metrics
function updateOverviewMetrics() {
    const uniqueMovies = [...new Set(movieData.map(item => item.movie_name))];
    const hitRate = (movieData.filter(item => item.label === 'Hit').length / movieData.length * 100);
    const avgEngagement = movieData.reduce((sum, item) => sum + item.engagement_rate, 0) / movieData.length;

    document.getElementById('total-movies').textContent = uniqueMovies.length;
    document.getElementById('total-datapoints').textContent = movieData.length.toLocaleString();
    document.getElementById('hit-rate').textContent = hitRate.toFixed(1) + '%';
    document.getElementById('avg-engagement').textContent = avgEngagement.toFixed(3);
}

// Create success distribution chart
function createSuccessDistributionChart() {
    const successCounts = {};
    movieData.forEach(item => {
        successCounts[item.label] = (successCounts[item.label] || 0) + 1;
    });

    const data = [{
        values: Object.values(successCounts),
        labels: Object.keys(successCounts),
        type: 'pie',
        marker: {
            colors: ['#27ae60', '#f39c12', '#e74c3c']
        },
        textinfo: 'label+percent',
        textposition: 'outside'
    }];

    const layout = {
        title: '',
        showlegend: true,
        height: 400,
        margin: { t: 0, b: 0, l: 0, r: 0 }
    };

    Plotly.newPlot('success-pie-chart', data, layout, { responsive: true });
}

// Create engagement chart
function createEngagementChart() {
    const engagementByCategory = {};

    movieData.forEach(item => {
        if (!engagementByCategory[item.label]) {
            engagementByCategory[item.label] = [];
        }
        engagementByCategory[item.label].push(item.engagement_rate);
    });

    const avgEngagement = {};
    Object.keys(engagementByCategory).forEach(category => {
        const rates = engagementByCategory[category];
        avgEngagement[category] = rates.reduce((sum, rate) => sum + rate, 0) / rates.length;
    });

    const data = [{
        x: Object.keys(avgEngagement),
        y: Object.values(avgEngagement),
        type: 'bar',
        marker: {
            color: ['#27ae60', '#f39c12', '#e74c3c']
        }
    }];

    const layout = {
        title: '',
        xaxis: { title: 'Success Category' },
        yaxis: { title: 'Average Engagement Rate' },
        height: 400,
        margin: { t: 20, b: 60, l: 60, r: 20 }
    };

    Plotly.newPlot('engagement-bar-chart', data, layout, { responsive: true });
}

// Create timeline chart
function createTimelineChart() {
    const timelineData = {};

    filteredData.forEach(item => {
        const key = `${item.label}`;
        if (!timelineData[key]) {
            timelineData[key] = { x: [], y: [] };
        }
        timelineData[key].x.push(item.days_before_release);
        timelineData[key].y.push(item.post_count);
    });

    const traces = Object.keys(timelineData).map(category => ({
        x: timelineData[category].x,
        y: timelineData[category].y,
        mode: 'markers',
        type: 'scatter',
        name: category,
        marker: {
            color: category === 'Hit' ? '#27ae60' : category === 'Average' ? '#f39c12' : '#e74c3c'
        }
    }));

    const layout = {
        title: '',
        xaxis: {
            title: 'Days Before Release',
            autorange: 'reversed'
        },
        yaxis: { title: 'Post Count' },
        height: 500,
        margin: { t: 20, b: 60, l: 60, r: 20 }
    };

    Plotly.newPlot('timeline-chart', traces, layout, { responsive: true });
}

// Create buzz type chart
function createBuzzTypeChart() {
    const buzzData = {};

    filteredData.forEach(item => {
        const key = `${item.buzz_type}-${item.label}`;
        buzzData[key] = (buzzData[key] || 0) + 1;
    });

    const buzzTypes = [...new Set(filteredData.map(item => item.buzz_type))];
    const categories = ['Hit', 'Average', 'Flop'];

    const traces = categories.map(category => ({
        x: buzzTypes,
        y: buzzTypes.map(type => buzzData[`${type}-${category}`] || 0),
        name: category,
        type: 'bar',
        marker: {
            color: category === 'Hit' ? '#27ae60' : category === 'Average' ? '#f39c12' : '#e74c3c'
        }
    }));

    const layout = {
        title: '',
        xaxis: { title: 'Buzz Type' },
        yaxis: { title: 'Count' },
        barmode: 'group',
        height: 400,
        margin: { t: 20, b: 60, l: 60, r: 20 }
    };

    Plotly.newPlot('buzz-type-chart', traces, layout, { responsive: true });
}

// Create engagement timeline chart
function createEngagementTimelineChart() {
    const engagementData = {};

    // Group by days and calculate average engagement
    filteredData.forEach(item => {
        const day = Math.floor(item.days_before_release / 30) * 30; // Group by 30-day periods
        const key = `${item.label}`;

        if (!engagementData[key]) {
            engagementData[key] = {};
        }

        if (!engagementData[key][day]) {
            engagementData[key][day] = [];
        }

        engagementData[key][day].push(item.engagement_rate);
    });

    const traces = Object.keys(engagementData).map(category => {
        const days = Object.keys(engagementData[category]).map(Number).sort((a, b) => b - a);
        const avgEngagement = days.map(day => {
            const rates = engagementData[category][day];
            return rates.reduce((sum, rate) => sum + rate, 0) / rates.length;
        });

        return {
            x: days,
            y: avgEngagement,
            mode: 'lines+markers',
            type: 'scatter',
            name: category,
            line: {
                color: category === 'Hit' ? '#27ae60' : category === 'Average' ? '#f39c12' : '#e74c3c'
            }
        };
    });

    const layout = {
        title: '',
        xaxis: {
            title: 'Days Before Release (grouped by 30-day periods)',
            autorange: 'reversed'
        },
        yaxis: { title: 'Average Engagement Rate' },
        height: 400,
        margin: { t: 20, b: 60, l: 60, r: 20 }
    };

    Plotly.newPlot('engagement-timeline-chart', traces, layout, { responsive: true });
}

// Cumulative buzz chart
function createCumulativeBuzzChart() {
    const categories = ['Hit', 'Average', 'Flop'];
    const traces = categories.map(cat => {
        const points = filteredData.filter(d => d.label === cat)
            .map(d => ({ day: d.days_before_release, posts: d.post_count }))
            .sort((a, b) => b.day - a.day);
        let cum = 0;
        const x = [], y = [];
        points.forEach(p => { cum += p.posts; x.push(p.day); y.push(cum); });
        return {
            x, y, mode: 'lines', type: 'scatter', name: cat,
            line: { color: cat === 'Hit' ? '#27ae60' : cat === 'Average' ? '#f39c12' : '#e74c3c' }
        };
    });
    Plotly.newPlot('cumulative-buzz-chart', traces, {
        xaxis: { title: 'Days Before Release', autorange: 'reversed' },
        yaxis: { title: 'Cumulative Posts' },
        margin: { t: 20, b: 60, l: 60, r: 20 }
    }, { responsive: true });
}
function buildLiveScoreScatter() {
    const el = document.getElementById('live-score-scatter');
    if (!el) return;
    const x = liveCompare.items.map(i => i.prediction_score ?? 0);
    const y = liveCompare.items.map(i => i.confidence ?? 0);
    const text = liveCompare.items.map(i => i.movie_name);
    const colors = liveCompare.items.map(i => colorByPrediction(i.prediction));
    Plotly.newPlot('live-score-scatter', [{ x, y, text, mode: 'markers', type: 'scatter', marker: { size: 12, color: colors } }], {
        xaxis: { title: 'Prediction Score' }, yaxis: { title: 'Confidence (%)', range: [0, 100] }, margin: { t: 20, b: 60, l: 60, r: 20 }
    }, { responsive: true });
}

// Populate movie selector
function populateMovieSelector() {
    const uniqueMovies = [...new Set(movieData.map(item => item.movie_name))];
    const selector = document.getElementById('movie-selector');

    selector.innerHTML = '';
    uniqueMovies.forEach((movie, index) => {
        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = movie;
        checkbox.checked = index < 3; // Check first 3 by default
        checkbox.addEventListener('change', updateMovieComparison);

        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(' ' + movie));
        selector.appendChild(label);
    });
}

// Update movie comparison
function updateMovieComparison() {
    const selectedMovies = Array.from(document.querySelectorAll('#movie-selector input:checked'))
        .map(cb => cb.value);

    if (selectedMovies.length === 0) return;

    // Movie buzz comparison
    const movieBuzzData = selectedMovies.map(movie => {
        const movieData_filtered = movieData.filter(item => item.movie_name === movie);
        const totalBuzz = movieData_filtered.reduce((sum, item) => sum + item.post_count, 0);
        const label = movieData_filtered[0]?.label || 'Unknown';

        return {
            movie: movie,
            buzz: totalBuzz,
            label: label
        };
    });

    const buzzTrace = {
        x: movieBuzzData.map(item => item.movie),
        y: movieBuzzData.map(item => item.buzz),
        type: 'bar',
        marker: {
            color: movieBuzzData.map(item =>
                item.label === 'Hit' ? '#27ae60' :
                    item.label === 'Average' ? '#f39c12' : '#e74c3c'
            )
        }
    };

    Plotly.newPlot('movie-buzz-chart', [buzzTrace], {
        title: '',
        xaxis: { title: 'Movie' },
        yaxis: { title: 'Total Buzz' },
        height: 400,
        margin: { t: 20, b: 60, l: 60, r: 20 }
    }, { responsive: true });

    // Movie engagement comparison
    const movieEngagementData = selectedMovies.map(movie => {
        const movieData_filtered = movieData.filter(item => item.movie_name === movie);
        const avgEngagement = movieData_filtered.reduce((sum, item) => sum + item.engagement_rate, 0) / movieData_filtered.length;
        const label = movieData_filtered[0]?.label || 'Unknown';

        return {
            movie: movie,
            engagement: avgEngagement,
            label: label
        };
    });

    const engagementTrace = {
        x: movieEngagementData.map(item => item.movie),
        y: movieEngagementData.map(item => item.engagement),
        type: 'bar',
        marker: {
            color: movieEngagementData.map(item =>
                item.label === 'Hit' ? '#27ae60' :
                    item.label === 'Average' ? '#f39c12' : '#e74c3c'
            )
        }
    };

    Plotly.newPlot('movie-engagement-chart', [engagementTrace], {
        title: '',
        xaxis: { title: 'Movie' },
        yaxis: { title: 'Average Engagement Rate' },
        height: 400,
        margin: { t: 20, b: 60, l: 60, r: 20 }
    }, { responsive: true });

    // Timeline comparison
    const timelineTraces = selectedMovies.map(movie => {
        const movieData_filtered = movieData.filter(item => item.movie_name === movie);
        const timelineData = {};

        movieData_filtered.forEach(item => {
            const day = Math.floor(item.days_before_release / 10) * 10; // Group by 10-day periods
            if (!timelineData[day]) {
                timelineData[day] = [];
            }
            timelineData[day].push(item.engagement_rate);
        });

        const days = Object.keys(timelineData).map(Number).sort((a, b) => b - a);
        const avgEngagement = days.map(day => {
            const rates = timelineData[day];
            return rates.reduce((sum, rate) => sum + rate, 0) / rates.length;
        });

        const label = movieData_filtered[0]?.label || 'Unknown';

        return {
            x: days,
            y: avgEngagement,
            mode: 'lines+markers',
            type: 'scatter',
            name: movie,
            line: {
                color: label === 'Hit' ? '#27ae60' : label === 'Average' ? '#f39c12' : '#e74c3c'
            }
        };
    });

    Plotly.newPlot('movie-timeline-comparison', timelineTraces, {
        title: '',
        xaxis: {
            title: 'Days Before Release',
            autorange: 'reversed'
        },
        yaxis: { title: 'Average Engagement Rate' },
        height: 500,
        margin: { t: 20, b: 60, l: 60, r: 20 }
    }, { responsive: true });
}

// Handle prediction
function handlePrediction(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const movieTitle = formData.get('movie-title') || document.getElementById('movie-title').value;
    const genre = document.getElementById('genre').value;
    const budget = parseInt(document.getElementById('budget').value);
    const socialBuzz = parseFloat(document.getElementById('social-buzz').value);
    const engagementRate = parseFloat(document.getElementById('engagement-rate-input').value);
    const daysBefore = parseInt(document.getElementById('days-before').value);

    // Simple prediction algorithm
    let score = 0;

    // Budget factor
    if (budget > 100) score += 0.3;
    else if (budget > 50) score += 0.2;
    else score += 0.1;

    // Social buzz factor
    score += socialBuzz * 0.4;

    // Engagement factor
    score += engagementRate * 0.3;

    // Genre factor
    const genreBonus = {
        'Action': 0.1,
        'Sci-Fi': 0.05,
        'Comedy': 0.05,
        'Drama': 0.02,
        'Horror': 0.03,
        'Romance': 0.01,
        'Thriller': 0.04
    };
    score += genreBonus[genre] || 0;

    // Determine prediction
    let prediction, confidence;
    if (score >= 0.7) {
        prediction = 'Hit';
        confidence = Math.min(95, 70 + (score - 0.7) * 100);
    } else if (score >= 0.4) {
        prediction = 'Average';
        confidence = Math.min(85, 60 + (score - 0.4) * 80);
    } else {
        prediction = 'Flop';
        confidence = Math.min(75, 50 + score * 50);
    }

    // Display results
    const resultsDiv = document.getElementById('prediction-results');
    const categoryBadge = document.getElementById('prediction-category');
    const confidenceScore = document.getElementById('confidence-score');
    const keyFactors = document.getElementById('key-factors');

    categoryBadge.textContent = prediction;
    categoryBadge.className = `prediction-badge ${prediction.toLowerCase()}`;
    confidenceScore.textContent = Math.round(confidence) + '%';

    // Key factors
    const factors = [
        { name: 'Social Media Buzz', impact: socialBuzz * 0.4 },
        { name: 'Engagement Rate', impact: engagementRate * 0.3 },
        { name: 'Budget Level', impact: budget > 100 ? 0.3 : budget > 50 ? 0.2 : 0.1 },
        { name: 'Genre Potential', impact: genreBonus[genre] || 0 },
        { name: 'Release Timing', impact: daysBefore < 30 ? 0.1 : daysBefore < 60 ? 0.05 : 0.02 }
    ];

    factors.sort((a, b) => b.impact - a.impact);

    keyFactors.innerHTML = '';
    factors.slice(0, 5).forEach(factor => {
        const li = document.createElement('li');
        li.textContent = `${factor.name}: ${(factor.impact * 100).toFixed(1)}% impact`;
        keyFactors.appendChild(li);
    });

    resultsDiv.style.display = 'block';
}

// Generate insights
function generateInsights() {
    const insights = [];

    // Calculate insights
    const hitMovies = movieData.filter(item => item.label === 'Hit');
    const flopMovies = movieData.filter(item => item.label === 'Flop');
    const averageMovies = movieData.filter(item => item.label === 'Average');

    if (hitMovies.length > 0 && flopMovies.length > 0) {
        const hitAvgEngagement = hitMovies.reduce((sum, item) => sum + item.engagement_rate, 0) / hitMovies.length;
        const flopAvgEngagement = flopMovies.reduce((sum, item) => sum + item.engagement_rate, 0) / flopMovies.length;

        insights.push({
            title: 'Engagement Impact',
            content: `Hit movies have ${(hitAvgEngagement / flopAvgEngagement).toFixed(1)}x higher engagement rates than flops (${hitAvgEngagement.toFixed(3)} vs ${flopAvgEngagement.toFixed(3)})`
        });

        const hitAvgPosts = hitMovies.reduce((sum, item) => sum + item.post_count, 0) / hitMovies.length;
        const flopAvgPosts = flopMovies.reduce((sum, item) => sum + item.post_count, 0) / flopMovies.length;

        insights.push({
            title: 'Social Media Buzz',
            content: `Hit movies generate ${(hitAvgPosts / flopAvgPosts).toFixed(1)}x more social media posts on average`
        });

        // Real movie financial insights (if available)
        const hitBudgets = hitMovies.filter(item => item.actual_budget > 0);
        const flopBudgets = flopMovies.filter(item => item.actual_budget > 0);

        if (hitBudgets.length > 0 && flopBudgets.length > 0) {
            const avgHitROI = hitBudgets.reduce((sum, item) => {
                return sum + (item.actual_revenue / item.actual_budget);
            }, 0) / hitBudgets.length;

            const avgFlopROI = flopBudgets.reduce((sum, item) => {
                return sum + (item.actual_revenue / item.actual_budget);
            }, 0) / flopBudgets.length;

            insights.push({
                title: 'Financial Performance',
                content: `Hit movies achieve ${avgHitROI.toFixed(1)}x return on investment vs ${avgFlopROI.toFixed(1)}x for flops`
            });
        }

        // Rating insights
        const hitRatings = hitMovies.filter(item => item.actual_rating > 0);
        const flopRatings = flopMovies.filter(item => item.actual_rating > 0);

        if (hitRatings.length > 0 && flopRatings.length > 0) {
            const avgHitRating = hitRatings.reduce((sum, item) => sum + item.actual_rating, 0) / hitRatings.length;
            const avgFlopRating = flopRatings.reduce((sum, item) => sum + item.actual_rating, 0) / flopRatings.length;

            insights.push({
                title: 'Quality vs Success',
                content: `Hit movies average ${avgHitRating.toFixed(1)}/10 rating compared to ${avgFlopRating.toFixed(1)}/10 for flops`
            });
        }
    }

    // Peak engagement timing
    const engagementByDay = {};
    movieData.forEach(item => {
        const dayGroup = Math.floor(item.days_before_release / 30) * 30;
        if (!engagementByDay[dayGroup]) {
            engagementByDay[dayGroup] = [];
        }
        engagementByDay[dayGroup].push(item.engagement_rate);
    });

    const avgEngagementByDay = Object.keys(engagementByDay).map(day => ({
        day: parseInt(day),
        avg: engagementByDay[day].reduce((sum, rate) => sum + rate, 0) / engagementByDay[day].length
    }));

    avgEngagementByDay.sort((a, b) => b.avg - a.avg);
    if (avgEngagementByDay.length > 0) {
        insights.push({
            title: 'Optimal Timing',
            content: `Peak engagement typically occurs ${avgEngagementByDay[0].day}-${avgEngagementByDay[0].day + 29} days before release`
        });
    }

    // Genre analysis
    const genreSuccess = {};
    movieData.forEach(item => {
        if (!genreSuccess[item.genre]) {
            genreSuccess[item.genre] = { total: 0, hits: 0 };
        }
        genreSuccess[item.genre].total++;
        if (item.label === 'Hit') {
            genreSuccess[item.genre].hits++;
        }
    });

    const genreRates = Object.keys(genreSuccess).map(genre => ({
        genre,
        rate: genreSuccess[genre].hits / genreSuccess[genre].total
    })).sort((a, b) => b.rate - a.rate);

    if (genreRates.length > 0) {
        insights.push({
            title: 'Genre Performance',
            content: `${genreRates[0].genre} movies have the highest success rate at ${(genreRates[0].rate * 100).toFixed(1)}%`
        });
    }

    // Display insights
    const container = document.getElementById('insights-content');
    container.innerHTML = '';

    insights.forEach(insight => {
        const card = document.createElement('div');
        card.className = 'insight-card';
        card.innerHTML = `
            <h4>${insight.title}</h4>
            <p>${insight.content}</p>
        `;
        container.appendChild(card);
    });
}

// Display data source information
function displayDataSourceInfo(dataSource) {
    const header = document.querySelector('.header-content');
    if (header) {
        const existingInfo = header.querySelector('.data-source-info');
        if (existingInfo) {
            existingInfo.remove();
        }

        const dataInfo = document.createElement('div');
        dataInfo.className = 'data-source-info';
        dataInfo.innerHTML = `
            <p style="margin-top: 0.5rem; opacity: 0.8; font-size: 0.9rem;">
                📊 Data Source: ${dataSource}
            </p>
        `;
        header.appendChild(dataInfo);
    }
}

// Utility functions
function showLoading() {
    document.getElementById('loading-spinner').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-spinner').style.display = 'none';
}

// Real Data Analysis Functions
function createRealDataAnalysis() {
    createBudgetRevenueChart();
    createRatingRevenueChart();
    createMoviePerformanceTable();
    createGenrePerformanceChart();
    createBudgetSuccessChart();
    buildTopRevenueChart();
    buildRoiHistogram();
    buildGenreSuccessHeatmap();
    buildCorrelationHeatmap();
    buildEngagementBubble();
    buildRoiRatingBubble();
    buildEngagementViolin();
    buildGenreSuccessSankey();
}

function buildCorrelationHeatmap() {
    const el = document.getElementById('corr-heatmap');
    if (!el) return;
    const unique = getUniqueMovies();
    if (unique.length === 0) return;
    const fields = [
        { key: 'engagement_rate', label: 'Engagement' },
        { key: 'post_count', label: 'Buzz' },
        { key: 'actual_rating', label: 'Rating' },
        { key: 'actual_budget', label: 'Budget' },
        { key: 'actual_revenue', label: 'Revenue' }
    ];
    const cols = fields.map(f => f.label);
    const data = fields.map(f => unique.map(m => Number(m[f.key]) || 0));
    const corr = cols.map((_, i) => cols.map((_, j) => pearson(data[i], data[j])));
    Plotly.newPlot('corr-heatmap', [{ z: corr, x: cols, y: cols, type: 'heatmap', colorscale: 'RdBu', zmin: -1, zmax: 1 }], {
        margin: { t: 20, b: 80, l: 80, r: 20 }
    }, { responsive: true });
}

function pearson(a, b) {
    const n = Math.min(a.length, b.length);
    if (n === 0) return 0;
    const ma = mean(a), mb = mean(b);
    let num = 0, da = 0, db = 0;
    for (let i = 0; i < n; i++) {
        const xa = a[i] - ma, xb = b[i] - mb;
        num += xa * xb; da += xa * xa; db += xb * xb;
    }
    return (da && db) ? num / Math.sqrt(da * db) : 0;
}
function mean(arr) { return arr.reduce((s, x) => s + x, 0) / (arr.length || 1); }

function buildEngagementBubble() {
    const el = document.getElementById('engagement-bubble-chart');
    if (!el) return;
    const unique = getUniqueMovies();
    const x = unique.map(m => m.post_count || 0);
    const y = unique.map(m => m.engagement_rate || 0);
    const size = unique.map(m => Math.max(8, Math.min(40, (m.actual_revenue || 0) / 1e7)));
    const color = unique.map(m => m.label === 'Hit' ? '#27ae60' : m.label === 'Average' ? '#f39c12' : '#e74c3c');
    Plotly.newPlot('engagement-bubble-chart', [{ x, y, mode: 'markers', type: 'scatter', marker: { size, color, sizemode: 'diameter' } }], {
        xaxis: { title: 'Buzz (Post Count)' }, yaxis: { title: 'Engagement Rate' }, margin: { t: 20, b: 60, l: 60, r: 20 }
    }, { responsive: true });
}

function buildRoiRatingBubble() {
    const el = document.getElementById('roi-rating-bubble');
    if (!el) return;
    const unique = getUniqueMovies().filter(m => (m.actual_revenue || 0) > 0 && (m.actual_budget || 0) > 0);
    const x = unique.map(m => m.actual_rating || 0);
    const y = unique.map(m => (m.actual_revenue || 0) / (m.actual_budget || 1));
    const size = unique.map(m => Math.max(8, Math.min(40, (m.actual_revenue || 0) / 1e7)));
    Plotly.newPlot('roi-rating-bubble', [{ x, y, mode: 'markers', type: 'scatter', marker: { size, color: '#2980b9' } }], {
        xaxis: { title: 'Rating (1-10)' }, yaxis: { title: 'ROI (x)' }, margin: { t: 20, b: 60, l: 60, r: 20 }
    }, { responsive: true });
}

function buildEngagementViolin() {
    const el = document.getElementById('engagement-violin');
    if (!el) return;
    const cats = ['Hit', 'Average', 'Flop'];
    const traces = cats.map(c => ({
        type: 'violin',
        y: movieData.filter(m => m.label === c).map(m => m.engagement_rate || 0),
        name: c,
        box: { visible: true },
        meanline: { visible: true },
        line: { color: c === 'Hit' ? '#27ae60' : c === 'Average' ? '#f39c12' : '#e74c3c' }
    }));
    Plotly.newPlot('engagement-violin', traces, { margin: { t: 20, b: 60, l: 60, r: 20 } }, { responsive: true });
}

function buildGenreSuccessSankey() {
    const el = document.getElementById('genre-success-sankey');
    if (!el) return;
    const unique = getUniqueMovies();
    const genres = [...new Set(unique.map(m => m.genre))];
    const labels = [...genres, 'Hit', 'Average', 'Flop'];
    const idx = (name) => labels.indexOf(name);
    const links = [];
    genres.forEach(g => {
        const total = unique.filter(m => m.genre === g).length;
        ['Hit', 'Average', 'Flop'].forEach(s => {
            const v = unique.filter(m => m.genre === g && m.label === s).length;
            if (v > 0) links.push({ source: idx(g), target: idx(s), value: v });
        });
    });
    const trace = { type: 'sankey',
        node: { label: labels },
        link: {
            source: links.map(l => l.source),
            target: links.map(l => l.target),
            value: links.map(l => l.value)
        }
    };
    Plotly.newPlot('genre-success-sankey', [trace], { margin: { t: 20, b: 20, l: 20, r: 20 } }, { responsive: true });
}

function buildTopRevenueChart() {
    const movies = getUniqueMovies().filter(m => (m.actual_revenue || 0) > 0);
    if (movies.length === 0 || !document.getElementById('top-revenue-chart')) return;
    const top = movies.sort((a, b) => (b.actual_revenue || 0) - (a.actual_revenue || 0)).slice(0, 10);
    const trace = {
        x: top.map(m => m.movie_name),
        y: top.map(m => (m.actual_revenue || 0) / 1e6),
        type: 'bar',
        marker: { color: '#34495e' },
        text: top.map(m => `$${((m.actual_revenue || 0) / 1e6).toFixed(0)}M`),
        textposition: 'auto'
    };
    Plotly.newPlot('top-revenue-chart', [trace], {
        yaxis: { title: 'Revenue (M USD)' },
        margin: { t: 10, b: 80, l: 60, r: 20 }
    }, { responsive: true });
}

function buildRoiHistogram() {
    const movies = getUniqueMovies().filter(m => (m.actual_revenue || 0) > 0 && (m.actual_budget || 0) > 0);
    if (movies.length === 0 || !document.getElementById('roi-histogram-chart')) return;
    const roi = movies.map(m => (m.actual_revenue || 0) / (m.actual_budget || 1));
    const trace = {
        x: roi,
        type: 'histogram',
        nbinsx: 20,
        marker: { color: '#8e44ad' }
    };
    Plotly.newPlot('roi-histogram-chart', [trace], {
        xaxis: { title: 'ROI (x)' },
        yaxis: { title: 'Count' },
        margin: { t: 10, b: 60, l: 60, r: 20 }
    }, { responsive: true });
}

function buildGenreSuccessHeatmap() {
    if (!document.getElementById('genre-success-heatmap')) return;
    const unique = getUniqueMovies();
    const genres = [...new Set(unique.map(m => m.genre))];
    const labels = ['Hit', 'Average', 'Flop'];

    const z = genres.map(g => labels.map(l => unique.filter(m => m.genre === g && m.label === l).length));
    const trace = {
        z, x: labels, y: genres, type: 'heatmap', colorscale: 'YlGnBu'
    };
    Plotly.newPlot('genre-success-heatmap', [trace], {
        margin: { t: 10, b: 60, l: 100, r: 20 }
    }, { responsive: true });
}

function createBudgetRevenueChart() {
    const moviesWithBudget = getUniqueMovies().filter(movie =>
        movie.actual_budget > 0 && movie.actual_revenue > 0
    );

    if (moviesWithBudget.length === 0) return;

    const trace = {
        x: moviesWithBudget.map(movie => movie.actual_budget / 1000000), // Convert to millions
        y: moviesWithBudget.map(movie => movie.actual_revenue / 1000000), // Convert to millions
        text: moviesWithBudget.map(movie => movie.movie_name),
        mode: 'markers',
        type: 'scatter',
        marker: {
            size: 12,
            color: moviesWithBudget.map(movie =>
                movie.label === 'Hit' ? '#27ae60' :
                    movie.label === 'Average' ? '#f39c12' : '#e74c3c'
            ),
            line: {
                width: 2,
                color: 'white'
            }
        },
        hovertemplate: '<b>%{text}</b><br>Budget: $%{x:.0f}M<br>Revenue: $%{y:.0f}M<extra></extra>'
    };

    // Add break-even line
    const maxBudget = Math.max(...moviesWithBudget.map(m => m.actual_budget / 1000000));
    const breakEvenLine = {
        x: [0, maxBudget],
        y: [0, maxBudget],
        mode: 'lines',
        type: 'scatter',
        name: 'Break Even',
        line: {
            dash: 'dash',
            color: 'gray'
        }
    };

    const layout = {
        title: '',
        xaxis: { title: 'Budget (Millions USD)' },
        yaxis: { title: 'Revenue (Millions USD)' },
        height: 400,
        margin: { t: 20, b: 60, l: 60, r: 20 },
        showlegend: false
    };

    Plotly.newPlot('budget-revenue-chart', [trace, breakEvenLine], layout, { responsive: true });
}

function createRatingRevenueChart() {
    const moviesWithRating = getUniqueMovies().filter(movie =>
        movie.actual_rating > 0 && movie.actual_revenue > 0
    );

    if (moviesWithRating.length === 0) return;

    const trace = {
        x: moviesWithRating.map(movie => movie.actual_rating),
        y: moviesWithRating.map(movie => movie.actual_revenue / 1000000),
        text: moviesWithRating.map(movie => movie.movie_name),
        mode: 'markers',
        type: 'scatter',
        marker: {
            size: 12,
            color: moviesWithRating.map(movie =>
                movie.label === 'Hit' ? '#27ae60' :
                    movie.label === 'Average' ? '#f39c12' : '#e74c3c'
            ),
            line: {
                width: 2,
                color: 'white'
            }
        },
        hovertemplate: '<b>%{text}</b><br>Rating: %{x}/10<br>Revenue: $%{y:.0f}M<extra></extra>'
    };

    const layout = {
        title: '',
        xaxis: { title: 'Rating (1-10)' },
        yaxis: { title: 'Revenue (Millions USD)' },
        height: 400,
        margin: { t: 20, b: 60, l: 60, r: 20 }
    };

    Plotly.newPlot('rating-revenue-chart', [trace], layout, { responsive: true });
}

function createMoviePerformanceTable() {
    const uniqueMovies = getUniqueMovies();

    if (uniqueMovies.length === 0) {
        document.getElementById('movie-performance-table').innerHTML = '<p>No movie data available</p>';
        return;
    }

    // Sort by revenue descending
    uniqueMovies.sort((a, b) => b.actual_revenue - a.actual_revenue);

    const tableHTML = `
        <table class="movie-table">
            <thead>
                <tr>
                    <th>Movie</th>
                    <th>Genre</th>
                    <th>Budget</th>
                    <th>Revenue</th>
                    <th>ROI</th>
                    <th>Rating</th>
                    <th>Success</th>
                </tr>
            </thead>
            <tbody>
                ${uniqueMovies.map(movie => {
        const budget = movie.actual_budget > 0 ? `$${(movie.actual_budget / 1000000).toFixed(0)}M` : 'N/A';
        const revenue = movie.actual_revenue > 0 ? `$${(movie.actual_revenue / 1000000).toFixed(0)}M` : 'N/A';
        const roi = movie.actual_budget > 0 && movie.actual_revenue > 0 ?
            `${(movie.actual_revenue / movie.actual_budget).toFixed(1)}x` : 'N/A';
        const rating = movie.actual_rating > 0 ? `${movie.actual_rating.toFixed(1)}/10` : 'N/A';

        return `
                        <tr>
                            <td><strong>${movie.movie_name}</strong></td>
                            <td>${movie.genre}</td>
                            <td>${budget}</td>
                            <td>${revenue}</td>
                            <td>${roi}</td>
                            <td>${rating}</td>
                            <td><span class="success-badge ${movie.label.toLowerCase()}">${movie.label}</span></td>
                        </tr>
                    `;
    }).join('')}
            </tbody>
        </table>
    `;

    document.getElementById('movie-performance-table').innerHTML = tableHTML;
}

function createGenrePerformanceChart() {
    const genreData = {};
    const uniqueMovies = getUniqueMovies();

    uniqueMovies.forEach(movie => {
        if (!genreData[movie.genre]) {
            genreData[movie.genre] = { total: 0, hits: 0, revenue: 0 };
        }
        genreData[movie.genre].total++;
        if (movie.label === 'Hit') {
            genreData[movie.genre].hits++;
        }
        genreData[movie.genre].revenue += movie.actual_revenue || 0;
    });

    const genres = Object.keys(genreData);
    const successRates = genres.map(genre =>
        (genreData[genre].hits / genreData[genre].total * 100)
    );

    const trace = {
        x: genres,
        y: successRates,
        type: 'bar',
        marker: {
            color: successRates.map(rate =>
                rate >= 60 ? '#27ae60' : rate >= 30 ? '#f39c12' : '#e74c3c'
            )
        },
        text: successRates.map(rate => `${rate.toFixed(1)}%`),
        textposition: 'auto',
        hovertemplate: '<b>%{x}</b><br>Success Rate: %{y:.1f}%<extra></extra>'
    };

    const layout = {
        title: '',
        xaxis: { title: 'Genre' },
        yaxis: { title: 'Success Rate (%)' },
        height: 400,
        margin: { t: 20, b: 60, l: 60, r: 20 }
    };

    Plotly.newPlot('genre-performance-chart', [trace], layout, { responsive: true });
}

function createBudgetSuccessChart() {
    const uniqueMovies = getUniqueMovies().filter(movie => movie.actual_budget > 0);

    if (uniqueMovies.length === 0) return;

    // Create budget ranges
    const budgetRanges = [
        { min: 0, max: 50, label: '<$50M' },
        { min: 50, max: 100, label: '$50-100M' },
        { min: 100, max: 200, label: '$100-200M' },
        { min: 200, max: 500, label: '>$200M' }
    ];

    const rangeData = budgetRanges.map(range => {
        const moviesInRange = uniqueMovies.filter(movie => {
            const budgetM = movie.actual_budget / 1000000;
            return budgetM >= range.min && budgetM < range.max;
        });

        const hits = moviesInRange.filter(movie => movie.label === 'Hit').length;
        const total = moviesInRange.length;

        return {
            range: range.label,
            successRate: total > 0 ? (hits / total * 100) : 0,
            total: total
        };
    }).filter(data => data.total > 0);

    const trace = {
        x: rangeData.map(data => data.range),
        y: rangeData.map(data => data.successRate),
        type: 'bar',
        marker: {
            color: rangeData.map(data =>
                data.successRate >= 60 ? '#27ae60' :
                    data.successRate >= 30 ? '#f39c12' : '#e74c3c'
            )
        },
        text: rangeData.map(data => `${data.successRate.toFixed(1)}%<br>(${data.total} movies)`),
        textposition: 'auto',
        hovertemplate: '<b>%{x}</b><br>Success Rate: %{y:.1f}%<extra></extra>'
    };

    const layout = {
        title: '',
        xaxis: { title: 'Budget Range' },
        yaxis: { title: 'Success Rate (%)' },
        height: 400,
        margin: { t: 20, b: 60, l: 60, r: 20 }
    };

    Plotly.newPlot('budget-success-chart', [trace], layout, { responsive: true });
}

function getUniqueMovies() {
    const movieMap = new Map();

    movieData.forEach(record => {
        if (!movieMap.has(record.movie_name)) {
            movieMap.set(record.movie_name, record);
        }
    });

    return Array.from(movieMap.values());
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `
        <div style="background: #e74c3c; color: white; padding: 1rem; border-radius: 5px; margin: 1rem;">
            <h3>Error</h3>
            <p>${message}</p>
        </div>
    `;
    document.body.appendChild(errorDiv);
}
