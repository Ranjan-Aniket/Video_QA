/**
 * Stage2Selection Component
 * 
 * Display ranked Gemini failures and select final 4 questions
 */

import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import reviewsAPI from '../api/reviewsApi';
import { GeminiFailure, PipelineCompleteEvent, WebSocketEvent } from '../types';

interface Stage2SelectionProps {
  videoId: string;
  reviewer: string;
  onComplete?: (excelPath: string) => void;
}

export function Stage2Selection({ videoId, reviewer, onComplete }: Stage2SelectionProps) {
  // ============================================================================
  // STATE
  // ============================================================================

  const [failures, setFailures] = useState<GeminiFailure[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // View mode: 'list' or 'comparison'
  const [viewMode, setViewMode] = useState<'list' | 'comparison'>('list');

  // ============================================================================
  // WEBSOCKET
  // ============================================================================

  const { isConnected } = useWebSocket({
    videoId,
    onEvent: (event: WebSocketEvent) => {
      if (event.type === 'pipeline_complete') {
        const completeEvent = event as PipelineCompleteEvent;
        onComplete?.(completeEvent.excel_path);
      }
    },
  });

  // ============================================================================
  // LOAD FAILURES
  // ============================================================================

  const loadFailures = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await reviewsAPI.getStage2Failures(videoId);
      setFailures(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load failures');
    } finally {
      setLoading(false);
    }
  }, [videoId]);

  useEffect(() => {
    loadFailures();
  }, [loadFailures]);

  // ============================================================================
  // SELECTION HANDLERS
  // ============================================================================

  const toggleSelection = (questionId: string) => {
    const newSelected = new Set(selectedIds);
    const numericId = parseInt(questionId);

    if (newSelected.has(numericId)) {
      newSelected.delete(numericId);
    } else {
      // Limit to 4 selections
      if (newSelected.size >= 4) {
        alert('You can only select 4 questions. Deselect one first.');
        return;
      }
      newSelected.add(numericId);
    }

    setSelectedIds(newSelected);
  };

  const clearSelection = () => {
    setSelectedIds(new Set());
  };

  // ============================================================================
  // SUBMIT SELECTION
  // ============================================================================

  const handleSubmit = async () => {
    if (selectedIds.size !== 4) {
      alert(`Please select exactly 4 questions (currently selected: ${selectedIds.size})`);
      return;
    }

    if (!confirm('Submit these 4 questions as your final selection?')) {
      return;
    }

    try {
      setSubmitting(true);
      const response = await reviewsAPI.submitStage2Selection(
        Array.from(selectedIds),
        reviewer
      );

      alert(
        `✅ Selection submitted successfully!\n\nExcel file ready:\n${response.excel_path}`
      );

      onComplete?.(response.excel_path);
    } catch (err: any) {
      alert(`Failed to submit selection: ${err.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  // ============================================================================
  // RENDER: LOADING/ERROR
  // ============================================================================

  if (loading) {
    return <div className="loading">Loading Gemini failures...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  if (failures.length === 0) {
    return (
      <div className="no-failures">
        <h2>No Gemini Failures Found</h2>
        <p>Gemini answered all questions correctly. No failures to select from.</p>
      </div>
    );
  }

  // ============================================================================
  // RENDER: MAIN UI
  // ============================================================================

  return (
    <div className="stage2-selection">
      {/* Header */}
      <header className="selection-header">
        <h1>Stage 2: Select Final 4 Questions</h1>
        <div className="connection-status">
          WebSocket: {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
        </div>
      </header>

      {/* Instructions */}
      <div className="instructions">
        <p>
          <strong>Select the 4 best Gemini failures</strong> from the ranked list below.
        </p>
        <p>
          Consider: failure score, severity, clarity, and educational value.
        </p>
      </div>

      {/* Selection Status */}
      <div className="selection-status">
        <div className="selected-count">
          <span className={selectedIds.size === 4 ? 'complete' : ''}>
            {selectedIds.size} / 4 selected
          </span>
        </div>
        <div className="view-controls">
          <button
            className={viewMode === 'list' ? 'active' : ''}
            onClick={() => setViewMode('list')}
          >
            📋 List View
          </button>
          <button
            className={viewMode === 'comparison' ? 'active' : ''}
            onClick={() => setViewMode('comparison')}
          >
            🔀 Comparison View
          </button>
        </div>
        <button onClick={clearSelection} disabled={selectedIds.size === 0}>
          Clear Selection
        </button>
        <button
          onClick={handleSubmit}
          disabled={selectedIds.size !== 4 || submitting}
          className="btn-primary"
        >
          {submitting ? 'Submitting...' : '✅ Submit Final Selection'}
        </button>
      </div>

      {/* List View */}
      {viewMode === 'list' && (
        <div className="failures-list">
          {failures.map((failure) => {
            const isSelected = selectedIds.has(parseInt(failure.question_id));

            return (
              <div
                key={failure.failure_id}
                className={`failure-card ${isSelected ? 'selected' : ''}`}
                onClick={() => toggleSelection(failure.question_id)}
              >
                {/* Rank Badge */}
                <div className="rank-badge">#{failure.rank}</div>

                {/* Selection Checkbox */}
                <div className="selection-checkbox">
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => {}}
                    onClick={(e) => e.stopPropagation()}
                  />
                </div>

                {/* Scores */}
                <div className="scores">
                  <div className="score-item">
                    <label>Combined Score</label>
                    <div className="score-value primary">
                      {failure.combined_score.toFixed(2)}
                    </div>
                  </div>
                  <div className="score-item">
                    <label>Failure Score</label>
                    <div className="score-value">{failure.failure_score.toFixed(2)}</div>
                  </div>
                  <div className="score-item">
                    <label>Severity</label>
                    <div className="score-value">{failure.severity_score.toFixed(2)}</div>
                  </div>
                  <div className="score-item">
                    <label>Clarity</label>
                    <div className="score-value">{failure.clarity_score.toFixed(2)}</div>
                  </div>
                </div>

                {/* Question */}
                <div className="question-section">
                  <label>Question</label>
                  <div className="question-text">{failure.question_text}</div>
                </div>

                {/* Answers */}
                <div className="answers-section">
                  <div className="answer-box correct">
                    <label>✅ Correct Answer</label>
                    <div className="answer-text">{failure.golden_answer}</div>
                  </div>
                  <div className="answer-box incorrect">
                    <label>❌ Gemini's Answer (Wrong)</label>
                    <div className="answer-text">{failure.gemini_answer}</div>
                  </div>
                </div>

                {/* Failure Type */}
                <div className="failure-type">
                  <span className="badge">{failure.failure_type}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Comparison View */}
      {viewMode === 'comparison' && (
        <div className="failures-comparison">
          <div className="comparison-grid">
            {failures.slice(0, 12).map((failure) => {
              const isSelected = selectedIds.has(parseInt(failure.question_id));

              return (
                <div
                  key={failure.failure_id}
                  className={`comparison-card ${isSelected ? 'selected' : ''}`}
                  onClick={() => toggleSelection(failure.question_id)}
                >
                  <div className="card-header">
                    <span className="rank">#{failure.rank}</span>
                    <span className="score">{failure.combined_score.toFixed(1)}</span>
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => {}}
                    />
                  </div>
                  <div className="card-question">{failure.question_text}</div>
                  <div className="card-answers">
                    <div className="answer correct">
                      ✅ {failure.golden_answer.slice(0, 50)}...
                    </div>
                    <div className="answer incorrect">
                      ❌ {failure.gemini_answer.slice(0, 50)}...
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Selected Summary */}
      {selectedIds.size > 0 && (
        <div className="selected-summary">
          <h3>Your Selection ({selectedIds.size}/4)</h3>
          <div className="selected-items">
            {Array.from(selectedIds).map((id) => {
              const failure = failures.find((f) => parseInt(f.question_id) === id);
              if (!failure) return null;

              return (
                <div key={id} className="selected-item">
                  <span className="rank">#{failure.rank}</span>
                  <span className="question">{failure.question_text.slice(0, 80)}...</span>
                  <span className="score">{failure.combined_score.toFixed(2)}</span>
                  <button onClick={() => toggleSelection(failure.question_id)}>✕</button>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default Stage2Selection;