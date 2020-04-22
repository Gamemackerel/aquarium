# frozen_string_literal: true

class JobSubmissionController < ApplicationController

  before_filter :signed_in_user

  # Planner GUI

  def index
    respond_to do |format|
      format.json { render json: Plan.list(current_user).reverse }
      format.html { render layout: 'aq2' }
    end
  end
end
