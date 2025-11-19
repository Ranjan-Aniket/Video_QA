/**
 * Reviews API
 *
 * API client for Stage 1 and Stage 2 review operations
 */

import api from './client';
import type { Question, ReviewProgress, GeminiFailure } from '../types';

// Stage 1 Review APIs
export const getStage1Questions = async (videoId: string): Promise<Question[]> => {
  const { data } = await api.get(`/reviews/stage1/${videoId}/questions`);
  return data;
};

export const getStage1Progress = async (videoId: string): Promise<ReviewProgress> => {
  const { data } = await api.get(`/reviews/stage1/${videoId}/progress`);
  return data;
};

export const approveQuestion = async (questionId: number, reviewer: string): Promise<void> => {
  await api.post(`/reviews/stage1/question/${questionId}/approve`, { reviewer });
};

export const rejectQuestion = async (
  questionId: number,
  reviewer: string,
  feedback: string
): Promise<void> => {
  await api.post(`/reviews/stage1/question/${questionId}/reject`, {
    reviewer,
    feedback,
  });
};

export const bulkApprove = async (
  questionIds: number[],
  reviewer: string
): Promise<void> => {
  await api.post('/reviews/stage1/bulk-approve', {
    question_ids: questionIds,
    reviewer,
  });
};

export const bulkReject = async (
  questionIds: number[],
  reviewer: string,
  feedback: string
): Promise<void> => {
  await api.post('/reviews/stage1/bulk-reject', {
    question_ids: questionIds,
    reviewer,
    feedback,
  });
};

export const regenerateQuestion = async (
  questionId: number,
  instructions?: string
): Promise<void> => {
  await api.post(`/reviews/stage1/question/${questionId}/regenerate`, {
    instructions,
  });
};

export const submitStage1 = async (videoId: string): Promise<void> => {
  await api.post(`/reviews/stage1/${videoId}/submit`);
};

// Stage 2 Selection APIs
export const getStage2Failures = async (videoId: string): Promise<GeminiFailure[]> => {
  const { data } = await api.get(`/reviews/stage2/${videoId}/failures`);
  return data;
};

export const selectQuestion = async (questionId: number): Promise<void> => {
  await api.post(`/reviews/stage2/question/${questionId}/select`);
};

export const deselectQuestion = async (questionId: number): Promise<void> => {
  await api.post(`/reviews/stage2/question/${questionId}/deselect`);
};

export const submitStage2 = async (videoId: string): Promise<void> => {
  await api.post(`/reviews/stage2/${videoId}/submit`);
};

export const submitStage2Selection = async (
  questionIds: number[],
  reviewer: string
): Promise<{ excel_path: string }> => {
  const { data } = await api.post('/reviews/stage2/submit-selection', {
    question_ids: questionIds,
    reviewer,
  });
  return data;
};

const reviewsAPI = {
  // Stage 1
  getStage1Questions,
  getStage1Progress,
  approveQuestion,
  rejectQuestion,
  bulkApprove,
  bulkReject,
  regenerateQuestion,
  submitStage1,

  // Stage 2
  getStage2Failures,
  selectQuestion,
  deselectQuestion,
  submitStage2,
  submitStage2Selection,
};

export default reviewsAPI;
