// ==============================================================================
// CREDITGUARD AI - FRONTEND CLIENT SCRIPT
// ==============================================================================

const API_BASE_URL = window.location.origin.includes("localhost") || window.location.origin.includes("127.0.0.1")
    ? "" // relative if hosted by FastAPI
    : "http://localhost:8000";

// Preset Demo Scenarios
const PRESETS = {
    low_risk: {
        applicant_name: "Nguyễn Văn An",
        age: 32,
        monthly_income: 25000000,
        requested_amount: 30000000,
        loan_tenor_months: 12,
        total_existing_debt: 5000000,
        cic_max_credit_group: 1,
        inquiries_last_3m: 0,
        credit_card_utilization: 20,
        non_bank_debt_ratio: 0
    },
    med_risk: {
        applicant_name: "Lê Thị Thảo",
        age: 26,
        monthly_income: 14000000,
        requested_amount: 40000000,
        loan_tenor_months: 12,
        total_existing_debt: 18000000,
        cic_max_credit_group: 2,
        inquiries_last_3m: 3,
        credit_card_utilization: 65,
        non_bank_debt_ratio: 30
    },
    high_risk: {
        applicant_name: "Phạm Quốc Tuấn",
        age: 24,
        monthly_income: 10000000,
        requested_amount: 60000000,
        loan_tenor_months: 12,
        total_existing_debt: 45000000,
        cic_max_credit_group: 3,
        inquiries_last_3m: 6,
        credit_card_utilization: 95,
        non_bank_debt_ratio: 70
    }
};

// ------------------------------------------------------------------------------
// INITIALIZATION
// ------------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
    checkApiHealth();
    setupCurrencyInputFormatters();
    setupFormSubmission();
});

// Check FastAPI status
async function checkApiHealth() {
    const statusText = document.getElementById("api-status-text");
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            statusText.textContent = `FastAPI Online (v${data.version})`;
            statusText.parentElement.style.borderColor = "rgba(16, 185, 129, 0.4)";
        } else {
            statusText.textContent = "FastAPI: Standalone Demo Mode";
        }
    } catch (e) {
        statusText.textContent = "FastAPI: Standalone Demo Mode";
    }
}

// ------------------------------------------------------------------------------
// TAB NAVIGATION
// ------------------------------------------------------------------------------
function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(panel => panel.classList.remove("active"));

    const targetBtn = document.getElementById(`tab-btn-${tabId}`);
    const targetPanel = document.getElementById(`tab-content-${tabId}`);

    if (targetBtn && targetPanel) {
        targetBtn.classList.add("active");
        targetPanel.classList.add("active");
    }
}

// ------------------------------------------------------------------------------
// PRESET INJECTION
// ------------------------------------------------------------------------------
function applyPreset(type) {
    const p = PRESETS[type];
    if (!p) return;

    document.getElementById("input-applicant-name").value = p.applicant_name;
    document.getElementById("input-age").value = p.age;
    document.getElementById("input-monthly-income").value = p.monthly_income;
    document.getElementById("input-requested-amount").value = p.requested_amount;
    document.getElementById("input-loan-tenor").value = p.loan_tenor_months;
    document.getElementById("input-total-debt").value = p.total_existing_debt;
    document.getElementById("input-cic-group").value = p.cic_max_credit_group;
    document.getElementById("input-inquiries").value = p.inquiries_last_3m;
    
    document.getElementById("input-card-util").value = p.credit_card_utilization;
    updateRangeLabel("card-util-val", p.credit_card_utilization + "%");

    document.getElementById("input-non-bank").value = p.non_bank_debt_ratio;
    updateRangeLabel("non-bank-val", p.non_bank_debt_ratio + "%");

    updateCurrencyHints();

    // Trigger auto-score for instant feedback
    submitApplication();
}

function updateRangeLabel(elementId, value) {
    const el = document.getElementById(elementId);
    if (el) el.textContent = value;
}

function updateCurrencyHints() {
    const formatVND = num => Number(num || 0).toLocaleString("vi-VN") + " VND";
    document.getElementById("hint-income").textContent = formatVND(document.getElementById("input-monthly-income").value);
    document.getElementById("hint-amount").textContent = formatVND(document.getElementById("input-requested-amount").value);
    document.getElementById("hint-debt").textContent = formatVND(document.getElementById("input-total-debt").value);
}

function setupCurrencyInputFormatters() {
    ["input-monthly-income", "input-requested-amount", "input-total-debt"].forEach(id => {
        const input = document.getElementById(id);
        if (input) {
            input.addEventListener("input", updateCurrencyHints);
        }
    });
}

// ------------------------------------------------------------------------------
// FORM SUBMISSION & API CALL
// ------------------------------------------------------------------------------
function setupFormSubmission() {
    const form = document.getElementById("loan-application-form");
    if (form) {
        form.addEventListener("submit", e => {
            e.preventDefault();
            submitApplication();
        });
    }
}

async function submitApplication() {
    const submitBtn = document.getElementById("btn-submit-scoring");
    const originalText = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span>Đang phân tích dữ liệu...</span>`;

    const payload = {
        applicant_name: document.getElementById("input-applicant-name").value,
        age: parseInt(document.getElementById("input-age").value),
        monthly_income: parseFloat(document.getElementById("input-monthly-income").value),
        requested_amount: parseFloat(document.getElementById("input-requested-amount").value),
        loan_tenor_months: parseInt(document.getElementById("input-loan-tenor").value),
        total_existing_debt: parseFloat(document.getElementById("input-total-debt").value),
        cic_max_credit_group: parseInt(document.getElementById("input-cic-group").value),
        inquiries_last_3m: parseInt(document.getElementById("input-inquiries").value),
        credit_card_utilization: parseFloat(document.getElementById("input-card-util").value),
        non_bank_debt_ratio: parseFloat(document.getElementById("input-non-bank").value)
    };

    let result = null;

    try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            result = await response.json();
        } else {
            result = fallbackClientScoring(payload);
        }
    } catch (e) {
        // If FastAPI server is not active, fallback to client-side risk engine
        result = fallbackClientScoring(payload);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
    }

    if (result) {
        renderScoringResults(result);
    }
}

// ------------------------------------------------------------------------------
// CLIENT-SIDE FALLBACK SCORING ENGINE
// ------------------------------------------------------------------------------
function fallbackClientScoring(data) {
    const installment = data.requested_amount / data.loan_tenor_months;
    const debtService = data.total_existing_debt * 0.05;
    const dti = (installment + debtService) / data.monthly_income;
    const lti = data.requested_amount / (data.monthly_income * 12);

    let rawRisk = 0.03
        + ((data.cic_max_credit_group - 1) * 0.22)
        + (Math.max(0, dti - 0.40) * 0.45)
        + (Math.min(0.18, data.inquiries_last_3m * 0.04))
        + (Math.max(0, (data.credit_card_utilization - 60) / 100 * 0.18))
        + ((data.non_bank_debt_ratio / 100) * 0.12);

    const pd = Math.min(0.96, Math.max(0.02, rawRisk));
    const score = Math.round(850 - (pd * 550));

    let decision = "APPROVED";
    let badgeColor = "#10b981";
    let reason = "Hồ sơ thỏa mãn toàn bộ tiêu chí tín dụng: Tỷ lệ DTI lành mạnh, lịch sử tín dụng CIC chuẩn.";

    if (pd >= 0.45 || data.cic_max_credit_group >= 3) {
        decision = "REJECTED";
        badgeColor = "#ef4444";
        reason = "Xác suất vỡ nợ vượt quá khẩu vị rủi ro. Khách hàng có nợ xấu CIC hoặc gánh nặng nợ DTI quá cao.";
    } else if (pd >= 0.20 || data.cic_max_credit_group === 2) {
        decision = "MANUAL REVIEW";
        badgeColor = "#f59e0b";
        reason = "Phát hiện tín hiệu rủi ro mức độ trung bình (nợ chú ý hoặc tra cứu nhiều). Cần cán bộ thẩm định bổ sung.";
    }

    return {
        applicant_name: data.applicant_name,
        probability_of_default: pd,
        credit_score: score,
        decision: decision,
        badge_color: badgeColor,
        decision_reason: reason,
        debt_to_income_ratio: dti,
        loan_to_income_ratio: lti,
        top_risk_factors: [
            {
                dimension: "Lịch sử nợ CIC",
                contribution_percent: Math.min(35, (data.cic_max_credit_group - 1) * 35 + 5),
                status: data.cic_max_credit_group > 1 ? "Nguy hiểm" : "Tốt",
                detail: `Nhóm nợ cao nhất: Nhóm ${data.cic_max_credit_group}`
            },
            {
                dimension: "Gánh nặng nợ (DTI)",
                contribution_percent: Math.min(30, Math.round(dti * 40)),
                status: dti > 0.50 ? "Cảnh báo" : "Tốt",
                detail: `Tỷ lệ nợ/thu nhập: ${(dti * 100).toFixed(1)}%`
            },
            {
                dimension: "Tần suất tra cứu CIC (3M)",
                contribution_percent: Math.min(20, Math.round(data.inquiries_last_3m * 6.5)),
                status: data.inquiries_last_3m >= 3 ? "Cao" : "Bình thường",
                detail: `${data.inquiries_last_3m} lần tra cứu gần đây`
            },
            {
                dimension: "Hạn mức thẻ tín dụng",
                contribution_percent: Math.round(data.credit_card_utilization * 0.15),
                status: data.credit_card_utilization > 75 ? "Căng thẳng" : "An toàn",
                detail: `Sử dụng hạn mức: ${data.credit_card_utilization}%`
            }
        ]
    };
}

// ------------------------------------------------------------------------------
// RENDER SCORING RESULTS TO DOM
// ------------------------------------------------------------------------------
function renderScoringResults(res) {
    document.getElementById("results-placeholder").style.display = "none";
    const content = document.getElementById("results-content");
    content.style.display = "block";

    // Header info
    document.getElementById("res-applicant-name").textContent = res.applicant_name || "Khách hàng";

    // Verdict Badge
    const badge = document.getElementById("res-verdict-badge");
    const badgeText = document.getElementById("res-verdict-text");
    badge.className = "verdict-pill";
    if (res.decision === "APPROVED") {
        badge.classList.add("approved");
    } else if (res.decision === "MANUAL REVIEW") {
        badge.classList.add("review");
    } else {
        badge.classList.add("rejected");
    }
    badgeText.textContent = res.decision;

    // PD Value and Animated Gauge Arc
    const pdPercent = (res.probability_of_default * 100).toFixed(1);
    document.getElementById("res-pd-val").textContent = `${pdPercent}%`;

    // SVG Gauge: circumference of half-circle (r=80) is ~251.2
    const totalLength = 251.2;
    const progress = Math.min(1, Math.max(0, res.probability_of_default));
    const offset = totalLength - (progress * totalLength);
    const arc = document.getElementById("gauge-fill-arc");
    arc.style.strokeDashoffset = offset;
    arc.style.stroke = res.badge_color;

    // Credit Score & Tier
    document.getElementById("res-credit-score").textContent = res.credit_score;
    const tierEl = document.getElementById("res-score-tier");
    if (res.credit_score >= 750) {
        tierEl.textContent = "Hạng AAA - Cực kỳ tin cậy";
        tierEl.className = "metric-desc text-success";
    } else if (res.credit_score >= 650) {
        tierEl.textContent = "Hạng AA - Rủi ro thấp";
        tierEl.className = "metric-desc text-success";
    } else if (res.credit_score >= 550) {
        tierEl.textContent = "Hạng BBB - Rủi ro trung bình";
        tierEl.className = "metric-desc text-amber";
    } else {
        tierEl.textContent = "Hạng Subprime - Rủi ro cao";
        tierEl.className = "metric-desc text-red";
    }

    // DTI Metric
    const dtiVal = (res.debt_to_income_ratio * 100).toFixed(1);
    document.getElementById("res-dti-val").textContent = `${dtiVal}%`;
    const dtiStatus = document.getElementById("res-dti-status");
    if (res.debt_to_income_ratio <= 0.40) {
        dtiStatus.textContent = "Rất an toàn (<40%)";
        dtiStatus.className = "metric-desc text-success";
    } else if (res.debt_to_income_ratio <= 0.55) {
        dtiStatus.textContent = "Mức trung bình (40-55%)";
        dtiStatus.className = "metric-desc text-amber";
    } else {
        dtiStatus.textContent = "Gánh nặng nợ cao (>55%)";
        dtiStatus.className = "metric-desc text-red";
    }

    // Decision Reason
    document.getElementById("res-decision-reason").textContent = res.decision_reason;

    // Risk Factor Breakdown Bars
    const listEl = document.getElementById("risk-factors-list");
    listEl.innerHTML = "";

    res.top_risk_factors.forEach(factor => {
        let badgeClass = "p-green";
        let barColor = "#10b981";
        if (factor.status === "Nguy hiểm" || factor.status === "Cao") {
            badgeClass = "p-red";
            barColor = "#ef4444";
        } else if (factor.status === "Cảnh báo" || factor.status === "Căng thẳng") {
            badgeClass = "p-amber";
            barColor = "#f59e0b";
        }

        const item = document.createElement("div");
        item.className = "factor-item";
        item.innerHTML = `
            <div class="factor-header">
                <span class="factor-name">${factor.dimension} (${factor.detail})</span>
                <span class="factor-status-badge ${badgeClass}">${factor.status}</span>
            </div>
            <div class="factor-bar-bg">
                <div class="factor-bar-fill" style="width: ${Math.min(100, factor.contribution_percent * 2.5)}%; background-color: ${barColor};"></div>
            </div>
        `;
        listEl.appendChild(item);
    });

    // Smooth scroll to view results on small screens
    if (window.innerWidth < 1024) {
        content.scrollIntoView({ behavior: "smooth" });
    }
}

// ------------------------------------------------------------------------------
// BATCH DEMO SIMULATION
// ------------------------------------------------------------------------------
function runSimulatedBatch() {
    const demoBatch = [
        { name: "Vũ Minh Anh", amount: 25000000, income: 19000000, cic: 1, dti: 0.15, pd: 0.06, score: 817, verdict: "APPROVED" },
        { name: "Đỗ Quốc Bảo", amount: 45000000, income: 16000000, cic: 1, dti: 0.28, pd: 0.12, score: 784, verdict: "APPROVED" },
        { name: "Trần Hồng Phúc", amount: 30000000, income: 14000000, cic: 1, dti: 0.22, pd: 0.09, score: 801, verdict: "APPROVED" },
        { name: "Ngô Quang Huy", amount: 70000000, income: 20000000, cic: 2, dti: 0.42, pd: 0.28, score: 696, verdict: "MANUAL REVIEW" },
        { name: "Bùi Thị Mai", amount: 35000000, income: 12000000, cic: 1, dti: 0.31, pd: 0.16, score: 762, verdict: "APPROVED" },
        { name: "Đinh Xuân Hoàng", amount: 80000000, income: 15000000, cic: 3, dti: 0.58, pd: 0.58, score: 531, verdict: "REJECTED" },
        { name: "Hoàng Gia Bảo", amount: 20000000, income: 11000000, cic: 1, dti: 0.20, pd: 0.08, score: 806, verdict: "APPROVED" },
        { name: "Nguyễn Thu Trang", amount: 50000000, income: 17000000, cic: 2, dti: 0.38, pd: 0.24, score: 718, verdict: "MANUAL REVIEW" },
        { name: "Phan Văn Hậu", amount: 15000000, income: 10000000, cic: 1, dti: 0.18, pd: 0.07, score: 811, verdict: "APPROVED" },
        { name: "Lê Khắc Tiệp", amount: 90000000, income: 18000000, cic: 4, dti: 0.65, pd: 0.72, score: 454, verdict: "REJECTED" },
    ];

    document.getElementById("batch-results-area").style.display = "block";
    const tbody = document.getElementById("batch-table-body");
    tbody.innerHTML = "";

    let approved = 0, review = 0, rejected = 0;

    demoBatch.forEach(item => {
        if (item.verdict === "APPROVED") approved++;
        else if (item.verdict === "MANUAL REVIEW") review++;
        else rejected++;

        let badgeClass = item.verdict === "APPROVED" ? "p-green" : item.verdict === "MANUAL REVIEW" ? "p-amber" : "p-red";

        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><strong>${item.name}</strong></td>
            <td>${(item.amount).toLocaleString("vi-VN")} đ</td>
            <td>${(item.income).toLocaleString("vi-VN")} đ</td>
            <td>Nhóm ${item.cic}</td>
            <td>${(item.dti * 100).toFixed(0)}%</td>
            <td>${(item.pd * 100).toFixed(1)}%</td>
            <td><strong>${item.score}</strong></td>
            <td><span class="p-badge ${badgeClass}">${item.verdict}</span></td>
        `;
        tbody.appendChild(tr);
    });

    document.getElementById("batch-total-count").textContent = demoBatch.length;
    document.getElementById("batch-approve-rate").textContent = `${Math.round(approved / demoBatch.length * 100)}%`;
    document.getElementById("batch-review-rate").textContent = `${Math.round(review / demoBatch.length * 100)}%`;
    document.getElementById("batch-reject-rate").textContent = `${Math.round(rejected / demoBatch.length * 100)}%`;
}
