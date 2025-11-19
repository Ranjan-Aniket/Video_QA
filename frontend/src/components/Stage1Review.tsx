/**
 * Stage1Review Component
 * 
 * Display all 30 questions for human review with bulk actions
 */

import { useState, useEffect, useMemo, useCallback } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import reviewsAPI from '../api/reviewsApi';
import {
  Question,
  ReviewProgress,
  ReviewFilters,
  SortConfig,
  Stage1ProgressEvent,
  WebSocketEvent,
} from '../types';

interface Stage1ReviewProps {
  videoId: string;
  reviewer: string;
  onComplete?: () => void;
}

export function Stage1Review({ videoId, reviewer, onComplete }: Stage1ReviewProps) {
  // ============================================================================
  // STATE
  // ============================================================================

  const [questions, setQuestions] = useState<Question[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [progress, setProgress] = useState<ReviewProgress | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filter & Sort state
  const [filters, setFilters] = useState<ReviewFilters>({
    status: 'all',
    tier: 'all',
    taskType: 'all',
    searchTerm: '',
  });
  const [sortConfig] = useState<SortConfig>({
    field: 'none',
    direction: 'asc',
  });

  // Modal state
  const [rejectModalOpen, setRejectModalOpen] = useState(false);
  const [rejectFeedback, setRejectFeedback] = useState('');
  const [rejectQuestionId, setRejectQuestionId] = useState<number | null>(null);

  // ============================================================================
  // WEBSOCKET
  // ============================================================================

  const { isConnected } = useWebSocket({
    videoId,
    onEvent: (event: WebSocketEvent) => {
      if (event.type === 'stage1_progress') {
        const progressEvent = event as Stage1ProgressEvent;
        setProgress({
          approved: progressEvent.approved,
          rejected: progressEvent.rejected,
          pending: progressEvent.pending,
          total: progressEvent.total,
          progress_percent: progressEvent.progress_percent,
        });
      } else if (event.type === 'stage1_complete') {
        onComplete?.();
      } else if (event.type === 'regeneration_complete') {
        // Refresh questions to show new regenerated question
        loadQuestions();
      }
    },
  });

  // ============================================================================
  // LOAD QUESTIONS
  // ============================================================================

  const loadQuestions = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await reviewsAPI.getStage1Questions(videoId);
      setQuestions(data);

      // Load progress
      const progressData = await reviewsAPI.getStage1Progress(videoId);
      setProgress(progressData);
    } catch (err: any) {
      setError(err.message || 'Failed to load questions');
    } finally {
      setLoading(false);
    }
  }, [videoId]);

  useEffect(() => {
    loadQuestions();
  }, [loadQuestions]);

  // ============================================================================
  // FILTERED & SORTED QUESTIONS
  // ============================================================================

  const filteredAndSortedQuestions = useMemo(() => {
    let filtered = [...questions];

    // Apply filters
    if (filters.status !== 'all') {
      filtered = filtered.filter((q) => q.stage1_review_status === filters.status);
    }

    if (filters.tier !== 'all') {
      filtered = filtered.filter((q) => q.generation_tier === filters.tier);
    }

    if (filters.taskType !== 'all') {
      filtered = filtered.filter((q) => q.task_type === filters.taskType);
    }

    if (filters.searchTerm) {
      const term = filters.searchTerm.toLowerCase();
      filtered = filtered.filter(
        (q) =>
          q.question_text.toLowerCase().includes(term) ||
          q.golden_answer.toLowerCase().includes(term)
      );
    }

    // Apply sorting
    if (sortConfig.field !== 'none') {
      filtered.sort((a, b) => {
        const aVal = a[sortConfig.field as keyof Question];
        const bVal = b[sortConfig.field as keyof Question];

        if (aVal == null) return 1;
        if (bVal == null) return -1;

        const comparison = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
        return sortConfig.direction === 'asc' ? comparison : -comparison;
      });
    }

    return filtered;
  }, [questions, filters, sortConfig]);

  // ============================================================================
  // SELECTION HANDLERS
  // ============================================================================

  const toggleSelection = (id: number) => {
    const newSelected = new Set(selectedIds);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedIds(newSelected);
  };

  const selectAll = () => {
    const allIds = new Set(filteredAndSortedQuestions.map((q) => q.id));
    setSelectedIds(allIds);
  };

  const clearSelection = () => {
    setSelectedIds(new Set());
  };

  // ============================================================================
  // ACTION HANDLERS
  // ============================================================================

  const handleApprove = async (questionId: number) => {
    try {
      await reviewsAPI.approveQuestion(questionId, reviewer);
      await loadQuestions();
    } catch (err: any) {
      alert(`Failed to approve: ${err.message}`);
    }
  };

  const handleRejectClick = (questionId: number) => {
    setRejectQuestionId(questionId);
    setRejectFeedback('');
    setRejectModalOpen(true);
  };

  const handleRejectSubmit = async () => {
    if (!rejectQuestionId || !rejectFeedback.trim()) {
      alert('Please provide feedback for rejection');
      return;
    }

    try {
      await reviewsAPI.rejectQuestion(rejectQuestionId, reviewer, rejectFeedback);
      setRejectModalOpen(false);
      setRejectQuestionId(null);
      setRejectFeedback('');
      await loadQuestions();
    } catch (err: any) {
      alert(`Failed to reject: ${err.message}`);
    }
  };

  const handleBulkApprove = async () => {
    if (selectedIds.size === 0) {
      alert('No questions selected');
      return;
    }

    if (!confirm(`Approve ${selectedIds.size} questions?`)) return;

    try {
      await reviewsAPI.bulkApprove(Array.from(selectedIds), reviewer);
      clearSelection();
      await loadQuestions();
    } catch (err: any) {
      alert(`Failed to bulk approve: ${err.message}`);
    }
  };

  const handleBulkReject = async () => {
    if (selectedIds.size === 0) {
      alert('No questions selected');
      return;
    }

    const feedback = prompt(`Provide feedback for rejecting ${selectedIds.size} questions:`);
    if (!feedback) return;

    try {
      await reviewsAPI.bulkReject(Array.from(selectedIds), reviewer, feedback);
      clearSelection();
      await loadQuestions();
    } catch (err: any) {
      alert(`Failed to bulk reject: ${err.message}`);
    }
  };

  const handleRegenerate = async (questionId: number) => {
    if (!confirm('Regenerate this question? This will create a new version.')) return;

    try {
      await reviewsAPI.regenerateQuestion(questionId);
      // WebSocket will trigger refresh via regeneration_complete event
    } catch (err: any) {
      alert(`Failed to regenerate: ${err.message}`);
    }
  };

  // ============================================================================
  // RENDER
  // ============================================================================

  if (loading) {
    return <div className="loading">Loading questions...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  return (
    <div className="stage1-review">
      {/* Header */}
      <header className="review-header">
        <h1>Stage 1: Review Questions</h1>
        <div className="connection-status">
          WebSocket: {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
        </div>
      </header>

      {/* Progress Bar */}
      {progress && (
        <div className="progress-section">
          <div className="progress-stats">
            <span className="stat approved">✅ Approved: {progress.approved}</span>
            <span className="stat rejected">❌ Rejected: {progress.rejected}</span>
            <span className="stat pending">⏳ Pending: {progress.pending}</span>
            <span className="stat total">📊 Total: {progress.total}</span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${progress.progress_percent}%` }}
            />
          </div>
          <div className="progress-percent">{progress.progress_percent.toFixed(1)}%</div>
        </div>
      )}

      {/* Filters */}
      <div className="filters">
        <select
          value={filters.status}
          onChange={(e) => setFilters({ ...filters, status: e.target.value as any })}
        >
          <option value="all">All Statuses</option>
          <option value="pending">Pending</option>
          <option value="approved">Approved</option>
          <option value="rejected">Rejected</option>
        </select>

        <select
          value={filters.tier}
          onChange={(e) => setFilters({ ...filters, tier: e.target.value as any })}
        >
          <option value="all">All Tiers</option>
          <option value="template">Template</option>
          <option value="llama">Llama</option>
          <option value="gpt4mini">GPT-4-mini</option>
        </select>

        <input
          type="text"
          placeholder="Search questions..."
          value={filters.searchTerm}
          onChange={(e) => setFilters({ ...filters, searchTerm: e.target.value })}
        />
      </div>

      {/* Bulk Actions */}
      <div className="bulk-actions">
        <button onClick={selectAll}>Select All ({filteredAndSortedQuestions.length})</button>
        <button onClick={clearSelection}>Clear Selection</button>
        <span className="selected-count">{selectedIds.size} selected</span>
        <button onClick={handleBulkApprove} disabled={selectedIds.size === 0}>
          ✅ Bulk Approve
        </button>
        <button onClick={handleBulkReject} disabled={selectedIds.size === 0}>
          ❌ Bulk Reject
        </button>
      </div>

      {/* Questions Table */}
      <div className="questions-table">
        <table>
          <thead>
            <tr>
              <th>
                <input type="checkbox" onChange={(e) => e.target.checked ? selectAll() : clearSelection()} />
              </th>
              <th>#</th>
              <th>Status</th>
              <th>Tier</th>
              <th>Question</th>
              <th>Answer</th>
              <th>Confidence</th>
              <th>Attempt</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredAndSortedQuestions.map((q, idx) => (
              <tr
                key={q.id}
                className={`
                  status-${q.stage1_review_status}
                  ${selectedIds.has(q.id) ? 'selected' : ''}
                  ${q.is_regeneration ? 'regenerated' : ''}
                `}
              >
                <td>
                  <input
                    type="checkbox"
                    checked={selectedIds.has(q.id)}
                    onChange={() => toggleSelection(q.id)}
                  />
                </td>
                <td>{idx + 1}</td>
                <td>
                  <span className={`badge status-${q.stage1_review_status}`}>
                    {q.stage1_review_status}
                  </span>
                </td>
                <td>
                  <span className={`badge tier-${q.generation_tier}`}>
                    {q.generation_tier}
                  </span>
                </td>
                <td className="question-text">{q.question_text}</td>
                <td className="answer-text">{q.golden_answer}</td>
                <td>{q.confidence_score?.toFixed(2) || 'N/A'}</td>
                <td>
                  {q.generation_attempt > 1 && (
                    <span className="attempt-badge">
                      Attempt {q.generation_attempt}/3
                    </span>
                  )}
                </td>
                <td className="actions">
                  {q.stage1_review_status === 'pending' && (
                    <>
                      <button
                        className="btn-approve"
                        onClick={() => handleApprove(q.id)}
                      >
                        ✅
                      </button>
                      <button
                        className="btn-reject"
                        onClick={() => handleRejectClick(q.id)}
                      >
                        ❌
                      </button>
                    </>
                  )}

                  {q.stage1_review_status === 'rejected' &&
                    q.generation_attempt < 3 && (
                      <button
                        className="btn-regenerate"
                        onClick={() => handleRegenerate(q.id)}
                      >
                        🔄 Regenerate
                      </button>
                    )}

                  {q.stage1_feedback && (
                    <span className="feedback-indicator" title={q.stage1_feedback}>
                      💬
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Reject Modal */}
      {rejectModalOpen && (
        <div className="modal-overlay" onClick={() => setRejectModalOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>Reject Question</h2>
            <p>Please provide feedback explaining why this question is being rejected:</p>
            <textarea
              value={rejectFeedback}
              onChange={(e) => setRejectFeedback(e.target.value)}
              placeholder="e.g., Missing visual cue, Too easy, Unclear phrasing..."
              rows={5}
            />
            <div className="modal-actions">
              <button onClick={() => setRejectModalOpen(false)}>Cancel</button>
              <button onClick={handleRejectSubmit} className="btn-primary">
                Submit Rejection
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Stage1Review;