const charts = {};
let cachedStudents = [];
let cachedHistory = [];
let analyticsLoaded = false;

const PALETTE = {
  indigo: "#4f46e5",
  indigoSoft: "rgba(79, 70, 229, 0.22)",
  emerald: "#10b981",
  emeraldSoft: "rgba(16, 185, 129, 0.22)",
  rose: "#f43f5e",
  roseSoft: "rgba(244, 63, 94, 0.22)",
  amber: "#f59e0b",
  teal: "#0d9488",
  cyan: "#06b6d4",
  violet: "#8b5cf6",
  slate: "#64748b",
};

const MULTI_COLORS = [
  "#4f46e5", "#0d9488", "#06b6d4", "#8b5cf6", "#f59e0b",
  "#10b981", "#ec4899", "#3b82f6", "#14b8a6", "#f97316",
];

function getChartTheme() {
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  return {
    text: isDark ? "#cbd5e1" : "#334155",
    grid: isDark ? "rgba(148, 163, 184, 0.14)" : "rgba(15, 23, 42, 0.08)",
    surface: isDark ? "#111827" : "#ffffff",
  };
}

function renderChart(canvasId, config) {
  const el = document.getElementById(canvasId);
  if (!el) return;
  if (charts[canvasId]) {
    charts[canvasId].destroy();
  }
  charts[canvasId] = new Chart(el.getContext("2d"), config);
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatDate(iso) {
  if (!iso) return "--";
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleString();
}

/* ==================== Navigation & Theme ==================== */
function switchTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-panel").forEach((p) => {
    p.classList.toggle("active", p.id === `tab-${tabName}`);
  });
  if (tabName === "analytics" && !analyticsLoaded) {
    loadAnalytics();
  }
  if (tabName === "history") {
    loadHistory();
  }
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

const themeBtn = document.getElementById("theme-toggle");
themeBtn.addEventListener("click", () => {
  const root = document.documentElement;
  const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
  root.setAttribute("data-theme", next);
  document.getElementById("theme-icon").textContent = next === "dark" ? "☀️" : "🌙";
  document.getElementById("theme-label").textContent = next === "dark" ? "Light Mode" : "Dark Mode";
  localStorage.setItem("spi-theme", next);
  if (analyticsLoaded) loadAnalytics();
});

(() => {
  const saved = localStorage.getItem("spi-theme");
  if (saved === "dark") {
    document.documentElement.setAttribute("data-theme", "dark");
    document.getElementById("theme-icon").textContent = "☀️";
    document.getElementById("theme-label").textContent = "Light Mode";
  }
})();

/* ==================== API Loaders ==================== */
async function apiFetch(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Request failed (${res.status})`);
  }
  return data;
}

async function checkHealth() {
  const pill = document.getElementById("api-status-pill");
  const txt = document.getElementById("api-status-text");
  try {
    const h = await apiFetch("/api/health");
    if (h.model_ready) {
      pill.className = "status-pill ready";
      txt.textContent = `Model Ready • ${h.total_dataset_records.toLocaleString()} rows`;
    } else {
      pill.className = "status-pill error";
      txt.textContent = "Dataset/Model Pending";
    }
  } catch {
    pill.className = "status-pill error";
    txt.textContent = "API Offline";
  }
}

async function loadSummaryAndMetrics() {
  try {
    const summary = await apiFetch("/api/summary");
    document.getElementById("kpi-students").textContent = summary.student_count;
    document.getElementById("kpi-predictions").textContent = summary.prediction_count;
    document.getElementById("kpi-pred-split").textContent =
      `${summary.placed_predictions} Placed / ${summary.not_placed_predictions} Not Placed`;
    document.getElementById("kpi-avg-cgpa").textContent = summary.average_cgpa.toFixed(2);

    const tbody = document.getElementById("recent-history-body");
    if (!summary.recent_history.length) {
      tbody.innerHTML = `<tr><td colspan="5" class="empty-cell">No predictions generated yet.</td></tr>`;
    } else {
      tbody.innerHTML = summary.recent_history
        .map(
          (r) => `
          <tr>
            <td><strong>${escapeHtml(r.student_name)}</strong></td>
            <td>${escapeHtml(r.programme)}</td>
            <td><span class="outcome-badge ${r.prediction === "Placed" ? "placed" : "not-placed"}">${escapeHtml(r.prediction)}</span></td>
            <td><code>${r.score.toFixed(1)}</code></td>
            <td><button class="btn btn-sm btn-ghost" onclick="openStudentReport(${r.student_id})">View</button></td>
          </tr>`
        )
        .join("");
    }
  } catch (err) {
    console.error("Summary error:", err);
  }

  try {
    const m = await apiFetch("/api/metrics");
    document.getElementById("kpi-roc-auc").textContent = (m.roc_auc * 100).toFixed(1) + "%";
    document.getElementById("kpi-dataset-size").textContent = `${m.total_records.toLocaleString()} synthetic records`;
    document.getElementById("m-accuracy").textContent = (m.accuracy * 100).toFixed(2) + "%";
    document.getElementById("m-precision").textContent = (m.precision * 100).toFixed(2) + "%";
    document.getElementById("m-recall").textContent = (m.recall * 100).toFixed(2) + "%";
    document.getElementById("m-f1").textContent = (m.f1_score * 100).toFixed(2) + "%";
    document.getElementById("m-roc").textContent = m.roc_auc.toFixed(4);
    document.getElementById("m-split").textContent = `${m.train_size.toLocaleString()} / ${m.test_size.toLocaleString()}`;
  } catch (err) {
    console.warn("Metrics unavailable:", err.message);
  }
}

async function loadStudents() {
  try {
    cachedStudents = await apiFetch("/api/students");
    renderStudentsTable(cachedStudents);
    populateStudentDropdown(cachedStudents);
  } catch (err) {
    console.error("Load students error:", err);
  }
}

function renderStudentsTable(list) {
  const tbody = document.getElementById("students-table-body");
  if (!list.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-cell">No matching student profiles.</td></tr>`;
    return;
  }
  tbody.innerHTML = list
    .map(
      (s) => `
      <tr>
        <td><strong>${escapeHtml(s.full_name)}</strong><br><small style="color:var(--text-soft)">${escapeHtml(s.email)}</small></td>
        <td>${escapeHtml(s.programme)}</td>
        <td><code>${s.cgpa.toFixed(2)}</code></td>
        <td>${escapeHtml(s.internship_experience)}</td>
        <td>${s.projects_completed}</td>
        <td style="white-space:nowrap">
          <button class="btn btn-sm btn-primary" onclick="openStudentReport(${s.id})">Predict</button>
          <button class="btn btn-sm btn-danger" onclick="deleteStudent(${s.id})">Delete</button>
        </td>
      </tr>`
    )
    .join("");
}

function populateStudentDropdown(list) {
  const select = document.getElementById("report-student-select");
  const prev = select.value;
  select.innerHTML = `<option value="">-- Choose a saved student --</option>` +
    list.map((s) => `<option value="${s.id}">${escapeHtml(s.full_name)} (${escapeHtml(s.programme)} • CGPA ${s.cgpa.toFixed(2)})</option>`).join("");
  if (prev && list.some((s) => String(s.id) === prev)) {
    select.value = prev;
  }
}

document.getElementById("student-search").addEventListener("input", (e) => {
  const q = e.target.value.trim().toLowerCase();
  const filtered = cachedStudents.filter(
    (s) =>
      s.full_name.toLowerCase().includes(q) ||
      s.programme.toLowerCase().includes(q) ||
      s.email.toLowerCase().includes(q)
  );
  renderStudentsTable(filtered);
});

/* ==================== Form Submit & Sample Data ==================== */
document.getElementById("fill-sample-btn").addEventListener("click", () => {
  const form = document.getElementById("student-form");
  const stamp = Math.floor(Math.random() * 900 + 100);
  form.full_name.value = `Priya Verma ${stamp}`;
  form.email.value = `priya.${stamp}@student.edu`;
  form.programme.value = "Artificial Intelligence and Data Science (AI & DS)";
  form.cgpa.value = "8.25";
  form.academic_performance.value = "8.1";
  form.attendance.value = "88";
  form.study_hours_per_week.value = "22";
  form.programming_skill.value = "8.0";
  form.internship_experience.value = "Yes";
  form.projects_completed.value = "4";
  form.certifications.value = "3";
  form.hackathons.value = "2";
  form.leadership_experience.value = "7.0";
  form.github_repositories.value = "7";
  form.linkedin_profile.value = "Yes";
  form.resume_score.value = "8.0";
  form.communication_skills.value = "7.5";
  form.teamwork.value = "8.2";
  form.problem_solving.value = "8.0";
  form.english_proficiency.value = "8.1";
  form.university_year.value = "4th Year";
  form.backlogs.value = "0";
});

document.getElementById("student-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const feedback = document.getElementById("form-feedback");
  feedback.textContent = "Saving profile...";
  feedback.className = "feedback-msg";

  const fd = new FormData(e.target);
  const payload = {
    full_name: fd.get("full_name"),
    email: fd.get("email"),
    programme: fd.get("programme"),
    university_year: fd.get("university_year"),
    cgpa: Number(fd.get("cgpa")),
    academic_performance: Number(fd.get("academic_performance")),
    attendance: Number(fd.get("attendance")),
    study_hours_per_week: Number(fd.get("study_hours_per_week")),
    backlogs: Number(fd.get("backlogs")),
    programming_skill: Number(fd.get("programming_skill")),
    internship_experience: fd.get("internship_experience"),
    projects_completed: Number(fd.get("projects_completed")),
    certifications: Number(fd.get("certifications")),
    hackathons: Number(fd.get("hackathons")),
    github_repositories: Number(fd.get("github_repositories")),
    linkedin_profile: fd.get("linkedin_profile"),
    resume_score: Number(fd.get("resume_score")),
    communication_skills: Number(fd.get("communication_skills")),
    teamwork: Number(fd.get("teamwork")),
    problem_solving: Number(fd.get("problem_solving")),
    english_proficiency: Number(fd.get("english_proficiency")),
    leadership_experience: Number(fd.get("leadership_experience")),
  };

  try {
    const created = await apiFetch("/api/students", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    feedback.textContent = `Saved profile for ${created.full_name}!`;
    feedback.className = "feedback-msg ok";
    e.target.reset();
    await Promise.all([loadStudents(), loadSummaryAndMetrics()]);
    analyticsLoaded = false;
  } catch (err) {
    feedback.textContent = err.message;
    feedback.className = "feedback-msg err";
  }
});

async function deleteStudent(sid) {
  if (!confirm("Delete this student profile and all its prediction records?")) return;
  try {
    await apiFetch(`/api/students/${sid}`, { method: "DELETE" });
    await Promise.all([loadStudents(), loadSummaryAndMetrics(), loadHistory()]);
    analyticsLoaded = false;
  } catch (err) {
    alert(err.message);
  }
}

/* ==================== Individual Prediction Report ==================== */
async function openStudentReport(sid) {
  switchTab("report");
  document.getElementById("report-student-select").value = String(sid);
  await generateReportForStudent(sid);
}

document.getElementById("run-predict-btn").addEventListener("click", () => {
  const sid = document.getElementById("report-student-select").value;
  if (!sid) {
    alert("Please select a student profile first.");
    return;
  }
  generateReportForStudent(Number(sid));
});

async function generateReportForStudent(sid) {
  const btn = document.getElementById("run-predict-btn");
  btn.disabled = true;
  btn.textContent = "Generating...";
  try {
    const rep = await apiFetch(`/api/predict/${sid}`, { method: "POST" });
    renderPredictionReport(rep);
    await loadSummaryAndMetrics();
    analyticsLoaded = false;
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate Placement Report";
  }
}

function renderPredictionReport(rep) {
  document.getElementById("report-empty").classList.add("hidden");
  document.getElementById("report-content").classList.remove("hidden");

  document.getElementById("rep-student-name").textContent = rep.student_name;
  document.getElementById("rep-programme").textContent = rep.programme;

  const badge = document.getElementById("rep-badge");
  badge.textContent = rep.prediction;
  badge.className = `outcome-badge ${rep.prediction === "Placed" ? "placed" : "not-placed"}`;
  document.getElementById("rep-score-val").textContent = rep.score.toFixed(1);

  const cov = rep.programme_coverage;
  const covBox = document.getElementById("rep-coverage-box");
  covBox.className = cov.is_low_coverage ? "coverage-box warn" : "coverage-box";
  document.getElementById("rep-coverage-count").textContent = `${cov.record_count.toLocaleString()} records`;
  document.getElementById("rep-coverage-note").textContent = cov.coverage_note;

  const theme = getChartTheme();

  // 6.1 Score Ring
  const ringColor = rep.prediction === "Placed" ? PALETTE.emerald : PALETTE.rose;
  renderChart("chart-rep-ring", {
    type: "doughnut",
    data: {
      labels: ["Model Score", "Remaining"],
      datasets: [
        {
          data: [rep.score, Math.max(0, 100 - rep.score)],
          backgroundColor: [ringColor, theme.grid],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "74%",
      plugins: { legend: { display: false } },
    },
  });

  // 6.3 What-If Scenarios
  const wif = rep.what_if || [];
  renderChart("chart-rep-whatif", {
    type: "bar",
    data: {
      labels: wif.length ? wif.map((w) => `${w.label} (${w.delta >= 0 ? "+" : ""}${w.delta} pts)`) : ["Profile already meets scenario targets"],
      datasets: [
        {
          label: "Current Score",
          data: wif.length ? wif.map((w) => w.current_score) : [rep.score],
          backgroundColor: PALETTE.slate,
          borderRadius: 6,
        },
        {
          label: "What-If Score",
          data: wif.length ? wif.map((w) => w.what_if_score) : [rep.score],
          backgroundColor: PALETTE.teal,
          borderRadius: 6,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { min: 0, max: 100, ticks: { color: theme.text }, grid: { color: theme.grid } },
        y: { ticks: { color: theme.text }, grid: { display: false } },
      },
      plugins: { legend: { labels: { color: theme.text } } },
    },
  });

  // 6.2 Student vs Benchmark (Normalised)
  const bm = rep.benchmarks || [];
  renderChart("chart-rep-benchmark", {
    type: "bar",
    data: {
      labels: bm.map((b) => b.label),
      datasets: [
        {
          label: "Student (Normalised %)",
          data: bm.map((b) => b.student_norm),
          backgroundColor: PALETTE.indigo,
          borderRadius: 5,
        },
        {
          label: "Dataset Median (Normalised %)",
          data: bm.map((b) => b.median_norm),
          backgroundColor: PALETTE.amber,
          borderRadius: 5,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { min: 0, max: 100, ticks: { color: theme.text }, grid: { color: theme.grid } },
        y: { ticks: { color: theme.text }, grid: { display: false } },
      },
      plugins: {
        legend: { labels: { color: theme.text } },
        tooltip: {
          callbacks: {
            afterLabel: (ctx) => {
              const row = bm[ctx.dataIndex];
              return `Exact: Student ${row.student_value} ${row.unit} | Median ${row.dataset_median} ${row.unit}`;
            },
          },
        },
      },
    },
  });

  // 6.4 Global Feature Importance
  const fi = rep.feature_importance || [];
  renderChart("chart-rep-importance", {
    type: "bar",
    data: {
      labels: fi.map((f) => f.feature),
      datasets: [
        {
          label: "Relative Importance (%)",
          data: fi.map((f) => f.importance),
          backgroundColor: PALETTE.violet,
          borderRadius: 6,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: theme.text }, grid: { color: theme.grid } },
        y: { ticks: { color: theme.text }, grid: { display: false } },
      },
      plugins: { legend: { display: false } },
    },
  });

  // 6.5 Programme Coverage
  const pc = rep.programme_counts || [];
  renderChart("chart-rep-coverage", {
    type: "bar",
    data: {
      labels: pc.map((p) => p.programme),
      datasets: [
        {
          label: "Training Records",
          data: pc.map((p) => p.count),
          backgroundColor: pc.map((p) => (p.is_selected ? PALETTE.rose : PALETTE.cyan)),
          borderRadius: 6,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: theme.text }, grid: { color: theme.grid } },
        y: { ticks: { color: theme.text, font: { size: 11 } }, grid: { display: false } },
      },
      plugins: { legend: { display: false } },
    },
  });

  // 6.6 Exact Benchmark Table
  document.getElementById("rep-benchmark-table").innerHTML = bm
    .map((b) => {
      const aboveMed = b.student_value >= b.dataset_median;
      return `
        <tr>
          <td><strong>${escapeHtml(b.label)}</strong></td>
          <td><code>${b.student_value} ${escapeHtml(b.unit)}</code></td>
          <td><code>${b.dataset_median} ${escapeHtml(b.unit)}</code></td>
          <td><code>${b.dataset_p75} ${escapeHtml(b.unit)}</code></td>
          <td><span class="tag ${aboveMed ? "tag-teal" : "tag-amber"}">${aboveMed ? "≥ Median" : "Below Median"}</span></td>
        </tr>`;
    })
    .join("");

  // 6.7 Recommendations
  document.getElementById("rep-recommendations").innerHTML = rep.recommendations
    .map(
      (r) => `
      <div class="rec-card ${escapeHtml(r.priority)}">
        <div class="rec-top">
          <h4>${escapeHtml(r.area)}</h4>
          <span class="tag">${escapeHtml(r.priority)} Priority</span>
        </div>
        <div class="rec-current">Current: ${escapeHtml(r.current_value)}</div>
        <p>${escapeHtml(r.message)}</p>
        <p><strong>Next step:</strong> ${escapeHtml(r.action_step)}</p>
      </div>`
    )
    .join("");
}

/* ==================== Analytics Dashboard (12 Charts) ==================== */
async function loadAnalytics() {
  try {
    const data = await apiFetch("/api/analytics");
    const ds = data.dataset;
    const app = data.application;
    const theme = getChartTheme();

    document.getElementById("an-dataset-total").textContent = `Dataset: ${ds.total_records.toLocaleString()} rows`;
    document.getElementById("an-app-total").textContent = `App Profiles: ${app.total_profiles} | Predictions: ${app.total_predictions}`;

    const barOpts = (horizontal = false, isRate = true) => ({
      indexAxis: horizontal ? "y" : "x",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          min: horizontal && isRate ? 0 : undefined,
          max: horizontal && isRate ? 100 : undefined,
          ticks: { color: theme.text },
          grid: { color: horizontal ? theme.grid : "transparent" },
        },
        y: {
          min: !horizontal && isRate ? 0 : undefined,
          max: !horizontal && isRate ? 100 : undefined,
          ticks: { color: theme.text },
          grid: { color: !horizontal ? theme.grid : "transparent" },
        },
      },
      plugins: { legend: { display: false } },
    });

    // 5.1 Distribution
    renderChart("an-dist", {
      type: "doughnut",
      data: {
        labels: ["Placed", "Not Placed"],
        datasets: [{
          data: [ds.placement_distribution.placed, ds.placement_distribution.not_placed],
          backgroundColor: [PALETTE.emerald, PALETTE.rose],
          borderWidth: 0,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "64%",
        plugins: { legend: { position: "bottom", labels: { color: theme.text } } },
      },
    });

    // 5.4 Internship
    renderChart("an-intern", {
      type: "bar",
      data: {
        labels: ds.by_internship.map((i) => `${i.label} (n=${i.count.toLocaleString()})`),
        datasets: [{
          label: "Placement Rate (%)",
          data: ds.by_internship.map((i) => i.rate),
          backgroundColor: [PALETTE.emerald, PALETTE.slate],
          borderRadius: 8,
        }],
      },
      options: barOpts(false, true),
    });

    // 5.3 CGPA Band
    renderChart("an-cgpa", {
      type: "bar",
      data: {
        labels: ds.by_cgpa_band.map((b) => b.band),
        datasets: [{
          label: "Placed %",
          data: ds.by_cgpa_band.map((b) => b.rate),
          backgroundColor: PALETTE.indigo,
          borderRadius: 7,
        }],
      },
      options: barOpts(false, true),
    });

    // 5.2 Programme Rate
    renderChart("an-prog-rate", {
      type: "bar",
      data: {
        labels: ds.by_programme_rate.map((p) => p.programme),
        datasets: [{
          label: "Placed %",
          data: ds.by_programme_rate.map((p) => p.rate),
          backgroundColor: PALETTE.teal,
          borderRadius: 6,
        }],
      },
      options: {
        ...barOpts(true, true),
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => {
                const item = ds.by_programme_rate[ctx.dataIndex];
                return `Records: ${item.count.toLocaleString()} (${item.placed_count.toLocaleString()} Placed)`;
              },
            },
          },
        },
      },
    });

    // 5.9 Programme Representation
    renderChart("an-prog-count", {
      type: "bar",
      data: {
        labels: ds.programme_representation.map((p) => p.programme),
        datasets: [{
          label: "Training Records",
          data: ds.programme_representation.map((p) => p.count),
          backgroundColor: PALETTE.cyan,
          borderRadius: 6,
        }],
      },
      options: barOpts(true, false),
    });

    // 5.5 Projects Line
    renderChart("an-projects", {
      type: "line",
      data: {
        labels: ds.by_projects.map((p) => p.label),
        datasets: [{
          label: "Placed %",
          data: ds.by_projects.map((p) => p.rate),
          borderColor: PALETTE.indigo,
          backgroundColor: PALETTE.indigoSoft,
          fill: true,
          tension: 0.32,
          pointRadius: 4,
        }],
      },
      options: barOpts(false, true),
    });

    // 5.6 Academic Performance
    renderChart("an-academic", {
      type: "bar",
      data: {
        labels: ds.by_academic_band.map((b) => b.band),
        datasets: [{
          label: "Placed %",
          data: ds.by_academic_band.map((b) => b.rate),
          backgroundColor: PALETTE.violet,
          borderRadius: 7,
        }],
      },
      options: barOpts(false, true),
    });

    // 5.7 Attendance
    renderChart("an-attendance", {
      type: "bar",
      data: {
        labels: ds.by_attendance_band.map((b) => b.band),
        datasets: [{
          label: "Placed %",
          data: ds.by_attendance_band.map((b) => b.rate),
          backgroundColor: PALETTE.amber,
          borderRadius: 7,
        }],
      },
      options: barOpts(false, true),
    });

    // 5.8 Programming Skill
    renderChart("an-programming", {
      type: "bar",
      data: {
        labels: ds.by_programming_band.map((b) => b.band),
        datasets: [{
          label: "Placed %",
          data: ds.by_programming_band.map((b) => b.rate),
          backgroundColor: PALETTE.emerald,
          borderRadius: 7,
        }],
      },
      options: barOpts(false, true),
    });

    // 5.10 App Profiles by Branch
    const branchData = app.profiles_by_branch.length
      ? app.profiles_by_branch
      : [{ programme: "No Saved Profiles", count: 1 }];
    renderChart("an-app-branch", {
      type: "doughnut",
      data: {
        labels: branchData.map((b) => b.programme),
        datasets: [{
          data: branchData.map((b) => b.count),
          backgroundColor: app.profiles_by_branch.length ? MULTI_COLORS : [theme.grid],
          borderWidth: 0,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "60%",
        plugins: { legend: { position: "bottom", labels: { color: theme.text } } },
      },
    });

    // 5.11 App CGPA Distribution
    renderChart("an-app-cgpa", {
      type: "bar",
      data: {
        labels: app.cgpa_distribution.map((c) => c.band),
        datasets: [{
          label: "Saved Students",
          data: app.cgpa_distribution.map((c) => c.count),
          backgroundColor: PALETTE.indigo,
          borderRadius: 7,
        }],
      },
      options: barOpts(false, false),
    });

    // 5.12 App Prediction Outcomes
    const hasPreds = app.prediction_outcomes.placed + app.prediction_outcomes.not_placed > 0;
    renderChart("an-app-outcomes", {
      type: "doughnut",
      data: {
        labels: hasPreds ? ["Placed", "Not Placed"] : ["No Predictions Yet"],
        datasets: [{
          data: hasPreds
            ? [app.prediction_outcomes.placed, app.prediction_outcomes.not_placed]
            : [1],
          backgroundColor: hasPreds ? [PALETTE.emerald, PALETTE.rose] : [theme.grid],
          borderWidth: 0,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "64%",
        plugins: { legend: { position: "bottom", labels: { color: theme.text } } },
      },
    });

    analyticsLoaded = true;
  } catch (err) {
    console.error("Analytics load failed:", err);
  }
}

/* ==================== Prediction History ==================== */
async function loadHistory() {
  try {
    cachedHistory = await apiFetch("/api/history");
    renderHistoryTable(cachedHistory);
  } catch (err) {
    console.error("History load failed:", err);
  }
}

function renderHistoryTable(list) {
  const tbody = document.getElementById("history-table-body");
  if (!list.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="empty-cell">No predictions recorded yet.</td></tr>`;
    return;
  }
  tbody.innerHTML = list
    .map(
      (h) => `
      <tr>
        <td><code>#${h.id}</code></td>
        <td>${escapeHtml(formatDate(h.created_at))}</td>
        <td><strong>${escapeHtml(h.student_name)}</strong></td>
        <td>${escapeHtml(h.programme)}</td>
        <td><code>${h.cgpa.toFixed(2)}</code></td>
        <td><span class="outcome-badge ${h.prediction === "Placed" ? "placed" : "not-placed"}">${escapeHtml(h.prediction)}</span></td>
        <td><code>${h.score.toFixed(1)} / 100</code></td>
        <td><button class="btn btn-sm btn-ghost" onclick="openStudentReport(${h.student_id})">Re-open</button></td>
      </tr>`
    )
    .join("");
}

document.getElementById("history-search").addEventListener("input", (e) => {
  const q = e.target.value.trim().toLowerCase();
  const filtered = cachedHistory.filter(
    (h) =>
      h.student_name.toLowerCase().includes(q) ||
      h.programme.toLowerCase().includes(q) ||
      h.prediction.toLowerCase().includes(q)
  );
  renderHistoryTable(filtered);
});

/* ==================== Initial Boot ==================== */
(async function init() {
  await checkHealth();
  await Promise.all([loadSummaryAndMetrics(), loadStudents(), loadHistory()]);
})();