document.addEventListener("DOMContentLoaded", () => {
    const homeTeam = document.getElementById("homeTeam");
    const awayTeam = document.getElementById("awayTeam");
    const predictButton = document.getElementById("predictButton");
    const buttonText = document.getElementById("buttonText");
    const loader = document.getElementById("loader");
    const errorMessage = document.getElementById("errorMessage");

    const resultSection = document.getElementById("resultSection");

    const predictionText = document.getElementById("predictionText");
    const matchText = document.getElementById("matchText");

    const homeProbability = document.getElementById("homeProbability");
    const drawProbability = document.getElementById("drawProbability");
    const awayProbability = document.getElementById("awayProbability");

    const homeBar = document.getElementById("homeBar");
    const drawBar = document.getElementById("drawBar");
    const awayBar = document.getElementById("awayBar");

    const homeGoals = document.getElementById("homeGoals");
    const awayGoals = document.getElementById("awayGoals");
    const totalGoals = document.getElementById("totalGoals");
    const predictedScore = document.getElementById("predictedScore");

    const reportHome = document.getElementById("reportHome");
    const reportAway = document.getElementById("reportAway");
    const reportPrediction = document.getElementById("reportPrediction");
    const h2hMatches = document.getElementById("h2hMatches");

    const historyContainer = document.getElementById("historyContainer");

    // -----------------------------------------
    // SCOREBOARD
    // -----------------------------------------

    const scoreHomeTeam = document.getElementById("scoreHomeTeam");
    const scoreAwayTeam = document.getElementById("scoreAwayTeam");

    const scoreHome = document.getElementById("scoreHome");
    const scoreAway = document.getElementById("scoreAway");

    // -----------------------------------------
    // TEAM BADGES
    // -----------------------------------------

    /*
       We first try to find the badge elements by ID.
       If the IDs are not present, we automatically
       find the two .team-crest-placeholder elements.
    */

    let homeCrest = document.getElementById("homeCrest");
    let awayCrest = document.getElementById("awayCrest");

    const crestElements =
        document.querySelectorAll(".team-crest-placeholder");

    if (!homeCrest && crestElements.length >= 1) {
        homeCrest = crestElements[0];
        homeCrest.id = "homeCrest";
    }

    if (!awayCrest && crestElements.length >= 2) {
        awayCrest = crestElements[1];
        awayCrest.id = "awayCrest";
    }

    // -----------------------------------------
    // INITIAL TEAM BADGES
    // -----------------------------------------

    updateTeamCrests();

    homeTeam.addEventListener("change", updateTeamCrests);
    awayTeam.addEventListener("change", updateTeamCrests);

    // -----------------------------------------
    // PREDICT MATCH
    // -----------------------------------------

    predictButton.addEventListener("click", async () => {
        const home = homeTeam.value;
        const away = awayTeam.value;

        errorMessage.textContent = "";
        errorMessage.classList.remove("show");

        if (!home || !away) {
            showError("Please select both teams.");
            return;
        }

        if (home === away) {
            showError("Home and away teams must be different.");
            return;
        }

        setLoading(true);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    home_team: home,
                    away_team: away
                })
            });

            const data = await response.json();

            if (!data.success) {
                throw new Error(
                    data.error || "Prediction failed."
                );
            }

            displayPrediction(data.result);

            await loadHistory();

        } catch (error) {
            showError(error.message);

        } finally {
            setLoading(false);
        }
    });

    // -----------------------------------------
    // DISPLAY PREDICTION
    // -----------------------------------------

    function displayPrediction(result) {

        const homeProb =
            Number(result.home_win_probability) * 100;

        const drawProb =
            Number(result.draw_probability) * 100;

        const awayProb =
            Number(result.away_win_probability) * 100;

        // -------------------------------------
        // MATCH INFORMATION
        // -------------------------------------

        matchText.textContent =
            `${result.home_team} vs ${result.away_team}`;

        // -------------------------------------
        // UPDATE TEAM BADGES
        // -------------------------------------

        updateCrest(homeCrest, result.home_team);
        updateCrest(awayCrest, result.away_team);

        // -------------------------------------
        // PREDICTION LABEL
        // -------------------------------------

        let predictionLabel = "DRAW";

        if (result.prediction === "H") {
            predictionLabel = "HOME WIN";
        } else if (result.prediction === "A") {
            predictionLabel = "AWAY WIN";
        }

        predictionText.textContent =
            predictionLabel;

        // -------------------------------------
        // SCOREBOARD TEAM NAMES
        // -------------------------------------

        if (scoreHomeTeam) {
            scoreHomeTeam.textContent =
                result.home_team;
        }

        if (scoreAwayTeam) {
            scoreAwayTeam.textContent =
                result.away_team;
        }

        // -------------------------------------
        // PREDICTED SCORE
        // -------------------------------------

        const homeScore =
            Math.round(
                Number(result.predicted_home_goals)
            );

        const awayScore =
            Math.round(
                Number(result.predicted_away_goals)
            );

        if (scoreHome) {
            scoreHome.textContent =
                homeScore;
        }

        if (scoreAway) {
            scoreAway.textContent =
                awayScore;
        }

        // Scoreboard reveal animation
        if (scoreHome) {
            scoreHome.classList.remove("score-reveal");
            void scoreHome.offsetWidth;
            scoreHome.classList.add("score-reveal");
        }

        if (scoreAway) {
            scoreAway.classList.remove("score-reveal");
            void scoreAway.offsetWidth;
            scoreAway.classList.add("score-reveal");
        }

        // -------------------------------------
        // PROBABILITY COUNTERS
        // -------------------------------------

        animateNumber(
            homeProbability,
            0,
            homeProb,
            900,
            "%"
        );

        animateNumber(
            drawProbability,
            0,
            drawProb,
            900,
            "%"
        );

        animateNumber(
            awayProbability,
            0,
            awayProb,
            900,
            "%"
        );

        // -------------------------------------
        // PROBABILITY BARS
        // -------------------------------------

        setTimeout(() => {

            homeBar.style.width =
                `${homeProb}%`;

            drawBar.style.width =
                `${drawProb}%`;

            awayBar.style.width =
                `${awayProb}%`;

        }, 100);

        // -------------------------------------
        // GOAL PREDICTIONS
        // -------------------------------------

        animateNumber(
            homeGoals,
            0,
            Number(result.predicted_home_goals),
            900,
            ""
        );

        animateNumber(
            awayGoals,
            0,
            Number(result.predicted_away_goals),
            900,
            ""
        );

        animateNumber(
            totalGoals,
            0,
            Number(result.predicted_total_goals),
            900,
            ""
        );

        predictedScore.textContent =
            result.predicted_score;

        // -------------------------------------
        // MATCH REPORT
        // -------------------------------------

        reportHome.textContent =
            result.home_team;

        reportAway.textContent =
            result.away_team;

        reportPrediction.textContent =
            predictionLabel;

        // -------------------------------------
        // H2H
        // -------------------------------------

        h2hMatches.textContent =
            "N/A";

        // -------------------------------------
        // SHOW RESULT
        // -------------------------------------

        resultSection.classList.remove(
            "hidden"
        );

        resultSection.classList.add(
            "show"
        );

        // -------------------------------------
        // RESULT ANIMATION
        // -------------------------------------

        resultSection.classList.remove(
            "result-reveal"
        );

        void resultSection.offsetWidth;

        resultSection.classList.add(
            "result-reveal"
        );

        // -------------------------------------
        // SCROLL TO RESULT
        // -------------------------------------

        setTimeout(() => {

            resultSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        }, 150);
    }

    // -----------------------------------------
    // TEAM CREST UPDATE
    // -----------------------------------------

    function updateTeamCrests() {

        if (homeCrest) {
            updateCrest(
                homeCrest,
                homeTeam.value
            );
        }

        if (awayCrest) {
            updateCrest(
                awayCrest,
                awayTeam.value
            );
        }
    }

    // -----------------------------------------
    // CREATE TEAM CREST
    // -----------------------------------------

    function updateCrest(element, teamName) {

        if (!element) {
            return;
        }

        if (!teamName) {
            element.textContent = "⚽";
            element.setAttribute(
                "aria-label",
                "No team selected"
            );

            element.classList.remove(
                "team-crest-active"
            );

            return;
        }

        const initials =
            getTeamInitials(teamName);

        element.textContent =
            initials;

        element.setAttribute(
            "aria-label",
            `${teamName} badge`
        );

        element.setAttribute(
            "title",
            teamName
        );

        element.classList.add(
            "team-crest-active"
        );

        // Restart badge animation
        element.classList.remove(
            "crest-pop"
        );

        void element.offsetWidth;

        element.classList.add(
            "crest-pop"
        );
    }

    // -----------------------------------------
    // GET TEAM INITIALS
    // -----------------------------------------

    function getTeamInitials(teamName) {

        if (!teamName) {
            return "⚽";
        }

        /*
           Remove country code such as:
           (ESP), (ENG), (GER)
        */

        const cleanName =
            teamName
                .replace(
                    /\s*\([A-Z]{3}\)\s*$/,
                    ""
                )
                .trim();

        const words =
            cleanName.split(/\s+/);

        // One-word team
        if (words.length === 1) {

            return words[0]
                .substring(0, 2)
                .toUpperCase();
        }

        // Two or more words
        return (
            words[0][0] +
            words[words.length - 1][0]
        ).toUpperCase();
    }

    // -----------------------------------------
    // ANIMATED NUMBER
    // -----------------------------------------

    function animateNumber(
        element,
        start,
        end,
        duration,
        suffix
    ) {

        if (!element) {
            return;
        }

        const startTime =
            performance.now();

        function update(currentTime) {

            const elapsed =
                currentTime - startTime;

            const progress =
                Math.min(
                    elapsed / duration,
                    1
                );

            // Smooth easing
            const eased =
                1 -
                Math.pow(
                    1 - progress,
                    3
                );

            const value =
                start +
                (end - start) *
                eased;

            if (suffix === "%") {

                element.textContent =
                    `${value.toFixed(1)}%`;

            } else {

                element.textContent =
                    value.toFixed(2);
            }

            if (progress < 1) {

                requestAnimationFrame(
                    update
                );
            }
        }

        requestAnimationFrame(
            update
        );
    }

    // -----------------------------------------
    // LOADING STATE
    // -----------------------------------------

    function setLoading(isLoading) {

        predictButton.disabled =
            isLoading;

        if (isLoading) {

            buttonText.textContent =
                "Analyzing Match...";

            loader.classList.add(
                "show"
            );

            predictButton.classList.add(
                "loading"
            );

        } else {

            buttonText.textContent =
                "Predict Match";

            loader.classList.remove(
                "show"
            );

            predictButton.classList.remove(
                "loading"
            );
        }
    }

    // -----------------------------------------
    // ERROR
    // -----------------------------------------

    function showError(message) {

        errorMessage.textContent =
            message;

        errorMessage.classList.add(
            "show"
        );
    }

    // -----------------------------------------
    // HISTORY
    // -----------------------------------------

    async function loadHistory() {

        try {

            const response =
                await fetch("/history");

            const data =
                await response.json();

            if (!data.success) {

                throw new Error(
                    data.error ||
                    "Could not load history."
                );
            }

            renderHistory(
                data.predictions
            );

        } catch (error) {

            historyContainer.innerHTML = `
                <p class="history-empty">
                    <span>⚠️</span>
                    Unable to load prediction history.
                </p>
            `;
        }
    }

    // -----------------------------------------
    // RENDER HISTORY
    // -----------------------------------------

    function renderHistory(
        predictions
    ) {

        if (
            !predictions ||
            predictions.length === 0
        ) {

            historyContainer.innerHTML = `
                <p class="history-empty">
                    <span>⚽</span>
                    No predictions yet.
                </p>
            `;

            return;
        }

        historyContainer.innerHTML =
            "";

        predictions.forEach(
            prediction => {

                const item =
                    document.createElement(
                        "div"
                    );

                item.className =
                    "history-item";

                let resultLabel =
                    "DRAW";

                if (
                    prediction.prediction === "H"
                ) {

                    resultLabel =
                        "HOME WIN";

                } else if (
                    prediction.prediction === "A"
                ) {

                    resultLabel =
                        "AWAY WIN";
                }

                item.innerHTML = `
                    <div class="history-match">
                        <strong>
                            ${escapeHtml(
                                prediction.home_team
                            )}
                        </strong>

                        <span>VS</span>

                        <strong>
                            ${escapeHtml(
                                prediction.away_team
                            )}
                        </strong>
                    </div>

                    <div class="history-details">

                        <div>
                            Result:
                            <strong>
                                ${resultLabel}
                            </strong>
                        </div>

                        <div>
                            Score:
                            <strong>
                                ${escapeHtml(
                                    prediction.predicted_score ||
                                    "N/A"
                                )}
                            </strong>
                        </div>

                        <div>
                            Total Goals:
                            <strong>
                                ${formatNumber(
                                    prediction.predicted_total_goals
                                )}
                            </strong>
                        </div>

                    </div>

                    <div class="history-date">
                        ${formatDate(
                            prediction.created_at
                        )}
                    </div>
                `;

                historyContainer.appendChild(
                    item
                );
            }
        );
    }

    // -----------------------------------------
    // FORMAT NUMBER
    // -----------------------------------------

    function formatNumber(value) {

        const number =
            Number(value);

        if (
            Number.isNaN(number)
        ) {

            return "N/A";
        }

        return number.toFixed(2);
    }

    // -----------------------------------------
    // FORMAT DATE
    // -----------------------------------------

    function formatDate(dateString) {

        if (!dateString) {

            return "Unknown time";
        }

        const date =
            new Date(dateString);

        if (
            Number.isNaN(
                date.getTime()
            )
        ) {

            return dateString;
        }

        return date.toLocaleString();
    }

    // -----------------------------------------
    // ESCAPE HTML
    // -----------------------------------------

    function escapeHtml(value) {

        if (
            value === null ||
            value === undefined
        ) {

            return "";
        }

        return String(value)
            .replace(
                /&/g,
                "&amp;"
            )
            .replace(
                /</g,
                "&lt;"
            )
            .replace(
                />/g,
                "&gt;"
            )
            .replace(
                /"/g,
                "&quot;"
            )
            .replace(
                /'/g,
                "&#039;"
            );
    }

    // -----------------------------------------
    // INITIAL HISTORY LOAD
    // -----------------------------------------

    loadHistory();
});