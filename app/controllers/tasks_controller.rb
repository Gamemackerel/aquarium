class TasksController < ApplicationController
  # GET /tasks
  # GET /tasks.json
  def index
    @task_prototype = TaskPrototype.find(params[:task_prototype_id])
    @tasks = @task_prototype.tasks

    respond_to do |format|
      format.html # index.html.erb
      format.json { render json: @tasks }
    end
  end

  # GET /tasks/1
  # GET /tasks/1.json
  def show
    @task = Task.find(params[:id])

    respond_to do |format|
      format.html # show.html.erb
      format.json { render json: @task }
    end
  end

  # GET /tasks/new
  # GET /tasks/new.json
  def new
    @task = Task.new
    @task.specification = TaskPrototype.find(params[:task_prototype_id].to_i).prototype
    @task.task_prototype_id = params[:task_prototype_id].to_i

    respond_to do |format|
      format.html # new.html.erb
      format.json { render json: @task }
    end
  end

  # GET /tasks/1/edit
  def edit
    @task = Task.find(params[:id])
  end

  # POST /tasks
  # POST /tasks.json
  def create
    @task = Task.new(params[:task])
    respond_to do |format|
      if @task.save
        format.html { redirect_to  tasks_url(task_prototype_id: @task.task_prototype.id), notice: "Task #{@task.name} was successfully created." }
        format.json { render json: @task, status: :created, location: @task }
      else
        format.html { render action: "new" }
        format.json { render json: @task.errors, status: :unprocessable_entity }
      end
    end
  end

  # PUT /tasks/1
  # PUT /tasks/1.json
  def update
    @task = Task.find(params[:id])
    respond_to do |format|
      if @task.update_attributes(params[:task])
        format.html { redirect_to tasks_url(task_prototype_id: @task.task_prototype.id), notice: 'Task was successfully updated.' }
        format.json { head :no_content }
      else
        format.html { render action: "edit" }
        format.json { render json: @task.errors, status: :unprocessable_entity }
      end
    end

  end

  # DELETE /tasks/1
  # DELETE /tasks/1.json
  def destroy
    @task = Task.find(params[:id])
    tpid = @task.task_prototype.id
    @task.destroy

    respond_to do |format|
      format.html { redirect_to tasks_url(task_prototype_id: tpid) }
      format.json { head :no_content }
    end
  end
end